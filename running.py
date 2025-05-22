import os
from typing import Dict, List, Optional, Union, Annotated, Literal, TypedDict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
import json
from datetime import datetime
from getpass import getpass
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FeedbackAspect(BaseModel):
    aspect: str
    sentiment: str
    score: float
    text: str

class CustomerFeedback(BaseModel):
    overall_sentiment: str = "neutral"
    aspects: List[FeedbackAspect] = []
    mixed_feedback: bool = False

class CompanyData(BaseModel):
    name: str
    faq_content: Optional[str] = None
    faq_path: Optional[str] = None
    product_categories: List[str]

# Define the state for our graph
class AgentState(TypedDict):
    company_data: Dict[str, Any]
    feedback_text: str
    faq_matches: List[str]
    sentiment_analysis: Optional[Dict[str, Any]]
    response: Optional[str]
    error: Optional[str]
    status: str

# Agent definitions
class SentimentAnalysisAgent:
    def __init__(self, llm):
        self.llm = llm
    
    async def analyze(self, state: AgentState) -> AgentState:
        """Analyze sentiment and identify different aspects of feedback"""
        company_name = state["company_data"]["name"]
        text = state["feedback_text"]
        
        if len(text.strip()) < 10:
            state["sentiment_analysis"] = CustomerFeedback(
                overall_sentiment="neutral",
                aspects=[FeedbackAspect(
                    aspect="general",
                    sentiment="neutral",
                    score=0.5,
                    text="Feedback too brief for detailed analysis"
                )],
                mixed_feedback=False
            ).model_dump()
            return state

        messages = [
            SystemMessage(content=f"""
            You are a sentiment analysis expert. Analyze the following feedback for {company_name}.
            Break down the feedback into distinct aspects and their sentiments.
            
            Format your response exactly as follows:
            OVERALL: [positive/negative/neutral/mixed]
            
            ASPECTS:
            [Aspect Name]: [positive/negative/neutral] - [brief explanation]
            """),
            HumanMessage(content=text)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content
            
            sections = response_text.split('\n\nASPECTS:')
            overall_part = sections[0].strip()
            aspects_part = sections[1].strip() if len(sections) > 1 else ""
            
            overall_sentiment = overall_part.replace('OVERALL:', '').strip().lower()
            
            aspects = []
            for line in aspects_part.split('\n'):
                if ':' in line and '-' in line:
                    aspect_name, rest = line.split(':', 1)
                    sentiment, explanation = rest.split('-', 1)
                    aspects.append(FeedbackAspect(
                        aspect=aspect_name.strip(),
                        sentiment=sentiment.strip().lower(),
                        score=0.8 if 'positive' in sentiment.lower() else 0.2,
                        text=explanation.strip()
                    ).model_dump())
            
            state["sentiment_analysis"] = CustomerFeedback(
                overall_sentiment=overall_sentiment,
                aspects=aspects,
                mixed_feedback=len(aspects) > 1
            ).model_dump()
            return state
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            state["error"] = f"Error analyzing feedback: {str(e)}"
            state["status"] = "error"
            return state

class FAQSearchAgent:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    async def search(self, state: AgentState) -> AgentState:
        """Search FAQ content for relevant matches"""
        query = state["feedback_text"]
        num_results = 2  # Default number of results
        
        try:
            search_results = self.vectorstore.similarity_search(query, k=num_results)
            state["faq_matches"] = [doc.page_content for doc in search_results]
            return state
        except Exception as e:
            logger.error(f"Error searching FAQ: {str(e)}")
            state["faq_matches"] = []
            return state

class ResponseGenerationAgent:
    def __init__(self, llm):
        self.llm = llm
    
    async def generate(self, state: AgentState) -> AgentState:
        """Generate appropriate customer service response"""
        try:
            company_name = state["company_data"]["name"]
            original_text = state["feedback_text"]
            feedback_analysis = state.get("sentiment_analysis", {})
            
            if not feedback_analysis or state.get("error"):
                state["response"] = "We appreciate your feedback. Our team will review it and get back to you soon."
                return state
            
            template = f"""You are a professional customer service representative for {company_name}.
            Write a response to the customer feedback below.
            Be professional, empathetic, and solution-focused.
            Address both positive and negative points specifically.
            Keep the response concise but thorough.
            
            Customer's message: {{feedback_text}}
            
            Sentiment analysis:
            {{feedback_analysis}}
            
            FAQ Information:
            {{faq_info}}
            """
            
            messages = [
                SystemMessage(content=template),
                HumanMessage(content="Please generate a response based on the above information.")
            ]
            
            faq_info = "\n\n".join(state.get("faq_matches", []))
            if not faq_info:
                faq_info = "No relevant FAQ matches found."
            
            response = await self.llm.ainvoke([
                SystemMessage(content=template.format(
                    feedback_text=original_text,
                    feedback_analysis=json.dumps(feedback_analysis, indent=2),
                    faq_info=faq_info
                )),
                HumanMessage(content="Please generate a response based on the above information.")
            ])
            
            state["response"] = response.content
            state["status"] = "success"
            return state
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            state["error"] = f"Error generating response: {str(e)}"
            state["status"] = "error"
            state["response"] = "We appreciate your feedback. Our team will review it and get back to you soon."
            return state

class ResponseSystem:
    def __init__(self, company_data: CompanyData):
        self.company_data = company_data
        self.faq_content = self.load_faq_content()
        self.setup_llm()
        self.setup_vectorstore()
        self.setup_agents()
        self.setup_graph()

    def load_faq_content(self) -> str:
        """Load FAQ content from either direct input or JSON file"""
        if self.company_data.faq_content:
            return self.company_data.faq_content
        elif self.company_data.faq_path:
            try:
                with open(self.company_data.faq_path, 'r') as f:
                    data = json.load(f)
                    faq_texts = [
                        f"Q: {faq['question']}\nA: {faq['answer']}"
                        for faq in data.get("faqs", [])
                    ]
                    return "\n\n".join(faq_texts)
            except Exception as e:
                logger.error(f"Error loading FAQ content: {str(e)}")
                raise ValueError(f"Failed to load FAQ content from {self.company_data.faq_path}")
        else:
            raise ValueError("Either faq_content or faq_path must be provided")

    def setup_llm(self):
        """Set up Groq LLM client"""
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            print("\nGroq API key not found in environment variables.")
            groq_api_key = getpass("Enter your Groq API key: ")
            os.environ["GROQ_API_KEY"] = groq_api_key
        
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="mistral-saba-24b",
            temperature=0.7,
            max_tokens=4096
        )

    def setup_vectorstore(self):
        """Set up vector store for FAQ search"""
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_text(self.faq_content)
        
        embeddings = HuggingFaceEmbeddings()
        
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

    def setup_agents(self):
        """Set up the agent system"""
        self.sentiment_agent = SentimentAnalysisAgent(self.llm)
        self.faq_agent = FAQSearchAgent(self.vectorstore)
        self.response_agent = ResponseGenerationAgent(self.llm)

    def setup_graph(self):
        """Set up the workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each step in the process
        workflow.add_node("analyze_sentiment", self.sentiment_agent.analyze)
        workflow.add_node("search_faq", self.faq_agent.search)
        workflow.add_node("generate_response", self.response_agent.generate)
        
        # Define the edges - the flow of the process
        workflow.add_edge("analyze_sentiment", "search_faq")
        workflow.add_edge("search_faq", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Set the entry point
        workflow.set_entry_point("analyze_sentiment")
        
        # Compile the graph
        self.graph = workflow.compile()

    async def process_feedback(self, feedback_text: str) -> Dict:
        """Main function to process customer feedback"""
        try:
            # Initialize the state
            initial_state = AgentState(
                company_data=self.company_data.model_dump(),
                feedback_text=feedback_text,
                faq_matches=[],
                sentiment_analysis=None,
                response=None,
                error=None,
                status="processing"
            )
            
            # Execute the graph
            result = await self.graph.ainvoke(initial_state)
            
            if result["status"] == "success":
                return {
                    "status": "success",
                    "analysis": result["sentiment_analysis"],
                    "response": result["response"],
                    "faq_matches": result["faq_matches"] if result["faq_matches"] else "No relevant FAQ matches found."
                }
            else:
                return {
                    "status": "error",
                    "message": result.get("error", "Unknown error occurred")
                }
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing feedback: {str(e)}"
            }

class FeedbackSystemController:
    """Controller class to manage the system's lifecycle"""
    
    def __init__(self):
        self.system = None
    
    async def initialize_system(self, company_data: CompanyData) -> bool:
        """Initialize the response system"""
        try:
            self.system = ResponseSystem(company_data)
            return True
        except Exception as e:
            logger.error(f"Error initializing system: {str(e)}")
            return False
    
    async def process_query(self, feedback_text: str) -> Dict:
        """Process a feedback query"""
        if not self.system:
            return {
                "status": "error",
                "message": "System not initialized"
            }
        
        return await self.system.process_feedback(feedback_text)

async def main():
    print("\n=== Enhanced Customer Feedback Analysis System ===")
    
    controller = FeedbackSystemController()
    
    print("\nPlease enter your company information.")
    company_name = input("Company Name: ").strip()
    
    # Ask user for FAQ input method
    faq_method = input("Would you like to (1) Enter FAQ content directly or (2) Load from JSON file? Enter 1 or 2: ").strip()
    
    faq_content = None
    faq_path = None
    
    if faq_method == "1":
        faq_entries = []
        print("\nEnter FAQ content (Type 'done' when finished):")
        while True:
            faq_entry = input("FAQ Entry (e.g., Q: Question? A: Answer.): ").strip()
            if faq_entry.lower() == 'done':
                break
            if faq_entry:
                faq_entries.append(faq_entry)
        faq_content = "\n\n".join(faq_entries)
    else:
        faq_path = input("Enter path to FAQ JSON file: ").strip()
    
    product_categories = input("\nEnter product categories (comma-separated): ").strip().split(',')

    if not company_name or (not faq_content and not faq_path) or not product_categories:
        print("\nCompany information is incomplete. Please try again.")
        return

    company_data = CompanyData(
        name=company_name,
        faq_content=faq_content,
        faq_path=faq_path,
        product_categories=[cat.strip() for cat in product_categories]
    )

    print("\nInitializing system...")
    if await controller.initialize_system(company_data):
        print("System initialized successfully!")
        
        while True:
            print("\n" + "=" * 50)
            feedback = input("Enter your feedback query (or 'quit' to exit): ").strip()
            
            if feedback.lower() == 'quit':
                print("Thank you for using the system. Goodbye!")
                break
            
            if not feedback:
                print("Please enter some feedback text.")
                continue
            
            print("\nProcessing feedback...")
            result = await controller.process_query(feedback)
            
            if result["status"] == "success":
                print("\n=== Sentiment Analysis ===")
                analysis = result["analysis"]
                print(f"Overall Sentiment: {analysis['overall_sentiment'].upper()}")
                print("\nDetailed Aspects:")
                for aspect in analysis['aspects']:
                    print(f"- {aspect['aspect']}: {aspect['sentiment']} ({aspect['text']})")
                
                print("\n=== Generated Response ===")
                print(result["response"])
                
                
                
                
                                                
                if isinstance(result["faq_matches"], list) and result["faq_matches"]:
                    print("\n=== Relevant FAQ Content ===")
                    for idx, match in enumerate(result["faq_matches"], 1):
                        print(f"\nMatch {idx}:")
                        print(match.strip())
                
                # After showing results, ask if user wants to continue
                continue_choice = input("\nDo you want to enter another query? (yes/no): ").strip().lower()
                if continue_choice != 'yes':
                    print("Thank you for using the system. Goodbye!")
                    break
            else:
                print("\nError:", result["message"])
                continue_choice = input("\nDo you want to enter another query? (yes/no): ").strip().lower()
                if continue_choice != 'yes':
                    print("Thank you for using the system. Goodbye!")
                    break
    else:
        print("\nSystem initialization failed. Please check the provided information and try again.")

# Add a function to construct sample FAQ data
def create_sample_faq_json(file_path: str):
    """Create a sample FAQ JSON file for testing"""
    sample_faqs = {
        "faqs": [
            {
                "question": "What are your shipping policies?",
                "answer": "We offer free shipping on orders over $50. Standard shipping takes 3-5 business days. Express shipping is available for an additional fee."
            },
            {
                "question": "How do I return a product?",
                "answer": "Returns can be initiated within 30 days of purchase. Please visit our returns portal on the website and follow the instructions. Return shipping is free for defective items."
            },
            {
                "question": "Do you offer international shipping?",
                "answer": "Yes, we ship to over 40 countries. International shipping rates vary by location. Please check our shipping calculator at checkout for exact costs."
            },
            {
                "question": "How can I track my order?",
                "answer": "Once your order ships, you'll receive a confirmation email with tracking information. You can also track your order by logging into your account on our website."
            },
            {
                "question": "What is your warranty policy?",
                "answer": "Most products come with a 1-year limited warranty against manufacturing defects. Premium products have extended warranties of up to 3 years."
            }
        ]
    }
    
    try:
        with open(file_path, 'w') as f:
            json.dump(sample_faqs, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error creating sample FAQ file: {str(e)}")
        return False

# Function to handle errors gracefully
def handle_system_error(error_message: str) -> None:
    """Display error message and handle system errors gracefully"""
    print(f"\n⚠️ ERROR: {error_message}")
    print("The system has encountered an error. Please try again or contact support.")
    logger.error(error_message)

# Advanced query processor with error handling
class AdvancedQueryProcessor:
    def __init__(self, controller: FeedbackSystemController):
        self.controller = controller
        self.history = []  # Store query history
    
    async def process_query_with_retry(self, query: str, max_retries: int = 2) -> Dict:
        """Process a query with automatic retry on failure"""
        attempts = 0
        while attempts <= max_retries:
            try:
                result = await self.controller.process_query(query)
                if result["status"] == "success":
                    # Add to history on success
                    self.history.append({
                        "timestamp": datetime.now().isoformat(),
                        "query": query,
                        "result": "success"
                    })
                    return result
                attempts += 1
                if attempts <= max_retries:
                    print(f"Processing failed, retrying ({attempts}/{max_retries})...")
            except Exception as e:
                attempts += 1
                logger.error(f"Error in query processing: {str(e)}")
                if attempts <= max_retries:
                    print(f"Error occurred, retrying ({attempts}/{max_retries})...")
        
        # Add failed attempt to history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result": "failed after retries"
        })
        
        return {
            "status": "error",
            "message": "Processing failed after multiple attempts"
        }
    
    def save_history(self, file_path: str) -> bool:
        """Save query history to a file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")
            return False

# Enhanced main function with additional features
async def enhanced_main():
    print("\n====================================")
    print("  Customer Feedback Analysis System  ")
    print("           Enhanced Edition          ")
    print("====================================\n")
    
    controller = FeedbackSystemController()
    
    try:
        # Offer sample data option
        print("Would you like to:")
        print("1. Enter company information manually")
        print("2. Use sample data for testing")
        setup_choice = input("Enter your choice (1/2): ").strip()
        
        if setup_choice == "2":
            # Setup with sample data
            print("\nSetting up system with sample data...")
            sample_faq_path = "sample_faq.json"
            
            if not os.path.exists(sample_faq_path):
                print("Creating sample FAQ data...")
                if not create_sample_faq_json(sample_faq_path):
                    print("Failed to create sample data. Switching to manual setup.")
                    setup_choice = "1"
            
            if setup_choice == "2":
                company_data = CompanyData(
                    name="Sample Company",
                    faq_path=sample_faq_path,
                    product_categories=["Electronics", "Home Goods", "Apparel"]
                )
                
                print("\nInitializing system with sample data...")
                if await controller.initialize_system(company_data):
                    print("System initialized successfully with sample data!")
                else:
                    print("Failed to initialize with sample data. Please try manual setup.")
                    return
        
        if setup_choice == "1":
            print("\nPlease enter your company information.")
            company_name = input("Company Name: ").strip()
            
            # Ask user for FAQ input method
            faq_method = input("Would you like to (1) Enter FAQ content directly or (2) Load from JSON file? Enter 1 or 2: ").strip()
            
            faq_content = None
            faq_path = None
            
            if faq_method == "1":
                faq_entries = []
                print("\nEnter FAQ content (Type 'done' when finished):")
                while True:
                    faq_entry = input("FAQ Entry (e.g., Q: Question? A: Answer.): ").strip()
                    if faq_entry.lower() == 'done':
                        break
                    if faq_entry:
                        faq_entries.append(faq_entry)
                faq_content = "\n\n".join(faq_entries)
            else:
                faq_path = input("Enter path to FAQ JSON file: ").strip()
            
            product_categories = input("\nEnter product categories (comma-separated): ").strip().split(',')

            if not company_name or (not faq_content and not faq_path) or not product_categories:
                print("\nCompany information is incomplete. Please try again.")
                return

            company_data = CompanyData(
                name=company_name,
                faq_content=faq_content,
                faq_path=faq_path,
                product_categories=[cat.strip() for cat in product_categories]
            )

            print("\nInitializing system...")
            if not await controller.initialize_system(company_data):
                print("\nSystem initialization failed. Please check the provided information and try again.")
                return
            
            print("System initialized successfully!")
        
        # Create advanced query processor
        query_processor = AdvancedQueryProcessor(controller)
        
        # Main processing loop
        while True:
            print("\n" + "=" * 50)
            print("Options:")
            print("1. Process customer feedback")
            print("2. Save query history")
            print("3. Exit")
            
            option = input("\nSelect an option (1-3): ").strip()
            
            if option == "3":
                print("Thank you for using the system. Goodbye!")
                break
            
            elif option == "2":
                history_path = input("Enter filename to save history: ").strip() or "query_history.json"
                if query_processor.save_history(history_path):
                    print(f"History saved successfully to {history_path}")
                else:
                    print("Failed to save history.")
                continue
            
            elif option == "1":
                feedback = input("Enter customer feedback (or 'back' to return to menu): ").strip()
                
                if feedback.lower() == 'back':
                    continue
                
                if not feedback:
                    print("Please enter some feedback text.")
                    continue
                
                print("\nProcessing feedback...")
                result = await query_processor.process_query_with_retry(feedback)
                
                if result["status"] == "success":
                    print("\n=== Sentiment Analysis ===")
                    analysis = result["analysis"]
                    print(f"Overall Sentiment: {analysis['overall_sentiment'].upper()}")
                    print("\nDetailed Aspects:")
                    for aspect in analysis['aspects']:
                        print(f"- {aspect['aspect']}: {aspect['sentiment']} ({aspect['text']})")
                    
                    print("\n=== Generated Response ===")
                    print(result["response"])
                    
                    if isinstance(result["faq_matches"], list) and result["faq_matches"]:
                        print("\n=== Relevant FAQ Content ===")
                        for idx, match in enumerate(result["faq_matches"], 1):
                            print(f"\nMatch {idx}:")
                            print(match.strip())
                else:
                    print("\nError:", result["message"])
            
            else:
                print("Invalid option. Please try again.")
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting gracefully...")
    except Exception as e:
        handle_system_error(str(e))

if __name__ == "__main__":
    import asyncio
    asyncio.run(enhanced_main())