"""
Agentic AI Workflow for GreenGov AI
Implements ProfileAgent, RetrievalAgent, EligibilityAgent, and ResponseAgent
"""

from typing import Dict, List, Tuple, Optional
from langchain.schema import Document
import google.generativeai as genai

class ProfileAgent:
    """Extracts user profile information from queries"""
    
    def __init__(self, model=None):
        self.model = model
        
    def extract_profile(self, query: str) -> Dict[str, any]:
        """Extract user details from natural language query"""
        
        # Simple keyword-based extraction
        profile = {
            "farmer_type": None,
            "state": None,
            "land_size": None,
            "goal": None,
            "original_query": query
        }
        
        # Farmer type detection
        farmer_keywords = {
            "small": ["small farmer", "marginal", "small-scale"],
            "large": ["large farmer", "commercial", "corporate"],
            "organic": ["organic", "natural farming", "chemical-free"]
        }
        
        for f_type, keywords in farmer_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                profile["farmer_type"] = f_type
                break
        
        # State extraction (common Indian states)
        states = [
            "punjab", "haryana", "uttar pradesh", "maharashtra", "gujarat",
            "rajasthan", "madhya pradesh", "bihar", "west bengal", "tamil nadu",
            "karnataka", "andhra pradesh", "telangana", "kerala", "odisha",
            "assam", "jharkhand", "chhattisgarh"
        ]
        
        for state in states:
            if state in query.lower():
                profile["state"] = state.title()
                break
        
        # Land size extraction (hectares/acres)
        import re
        land_patterns = [
            r'(\d+(?:\.\d+)?)\s*(hectares?|ha)',
            r'(\d+(?:\.\d+)?)\s*(acres?|ac)'
        ]
        
        for pattern in land_patterns:
            match = re.search(pattern, query.lower())
            if match:
                profile["land_size"] = f"{match.group(1)} {match.group(2)}"
                break
        
        # Goal extraction
        goals = {
            "solar_power": ["solar", "renewable energy", "electricity", "power", "energy"],
            "water_conservation": ["water", "irrigation", "drip", "sprinkler", "conservation"],
            "crop_insurance": ["insurance", "crop loss", "damage", "protection"],
            "soil_health": ["soil", "fertility", "nutrient", "fertilizer"]
        }
        
        for goal, keywords in goals.items():
            if any(keyword in query.lower() for keyword in keywords):
                profile["goal"] = goal.replace("_", " ").title()
                break
        
        return profile
    
    def format_profile_prompt(self, profile: Dict) -> str:
        """Format profile for prompt injection"""
        parts = ["User Profile:"]
        if profile["farmer_type"]:
            parts.append(f"- Farmer Type: {profile['farmer_type']}")
        if profile["state"]:
            parts.append(f"- State: {profile['state']}")
        if profile["land_size"]:
            parts.append(f"- Land Size: {profile['land_size']}")
        if profile["goal"]:
            parts.append(f"- Goal: {profile['goal']}")
        
        return "\n".join(parts)

class RetrievalAgent:
    """Handles retrieval of relevant scheme information"""
    
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
    
    def retrieve(self, query: str, profile: Dict, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve relevant chunks based on query and profile context"""
        
        # Enhance query with profile information
        enhanced_query = query
        if profile["state"]:
            enhanced_query += f" in {profile['state']}"
        if profile["goal"]:
            enhanced_query += f" for {profile['goal']}"
        
        chunks = self.rag_pipeline.retrieve_relevant_chunks(enhanced_query, k=k)
        
        # Organize by scheme
        schemes_info = {}
        for doc, score in chunks:
            scheme = doc.metadata.get("scheme_name", "Unknown")
            if scheme not in schemes_info:
                schemes_info[scheme] = []
            schemes_info[scheme].append((doc, score))
        
        return schemes_info, chunks

class EligibilityAgent:
    """Analyzes eligibility based on user profile and scheme documents"""
    
    def __init__(self, model):
        self.model = model
    
    def analyze_eligibility(self, profile: Dict, schemes_info: Dict) -> Dict:
        """Analyze eligibility for each scheme"""
        
        eligibility_results = {}
        
        for scheme_name, docs in schemes_info.items():
            # Extract key criteria from documents
            context = "\n".join([doc.page_content[:500] for doc, _ in docs[:3]])
            
            prompt = f"""
            Based on the scheme document below, analyze if the user is eligible for {scheme_name}.
            
            User Profile:
            - Farmer Type: {profile.get('farmer_type', 'Not specified')}
            - State: {profile.get('state', 'Not specified')}
            - Land Size: {profile.get('land_size', 'Not specified')}
            - Goal: {profile.get('goal', 'Not specified')}
            
            Scheme Document Excerpt:
            {context}
            
            Analyze eligibility and provide:
            1. Eligibility Status (Eligible/Partially Eligible/Not Eligible)
            2. Key criteria that match or don't match
            3. Specific requirements the user needs to meet
            
            Keep response concise and actionable.
            """
            
            try:
                response = self.model.generate_content(prompt)
                eligibility_results[scheme_name] = response.text
            except Exception as e:
                eligibility_results[scheme_name] = f"Eligibility analysis unavailable: {str(e)}"
        
        return eligibility_results

class ResponseAgent:
    """Generates final comprehensive response for the user"""
    
    def __init__(self, model):
        self.model = model
    
    def generate_response(self, query: str, profile: Dict, schemes_info: Dict, 
                          eligibility_results: Dict) -> str:
        """Generate final response with recommended schemes and details"""
        
        # Format context
        context_parts = []
        for scheme_name, docs in schemes_info.items():
            context_parts.append(f"\n=== {scheme_name} ===\n")
            for doc, score in docs[:2]:  # Top 2 chunks per scheme
                context_parts.append(f"Content: {doc.page_content[:800]}")
                context_parts.append(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
        
        context = "\n".join(context_parts)
        
        # Format eligibility
        eligibility_text = "\n".join([f"**{scheme}**:\n{analysis}" 
                                      for scheme, analysis in eligibility_results.items()])
        
        prompt = f"""
        You are GreenGov AI, an expert assistant for government sustainability schemes in India.
        
        User Question: {query}
        
        {ProfileAgent().format_profile_prompt(profile)}
        
        RELEVANT SCHEME DOCUMENTS:
        {context}
        
        ELIGIBILITY ANALYSIS:
        {eligibility_text}
        
        Please provide a comprehensive response with the following structure:
        
        ## 🌿 Recommended Schemes
        (List top 2-3 most relevant schemes)
        
        ## ✅ Eligibility Analysis
        (Summarize eligibility for each recommended scheme)
        
        ## 💰 Benefits
        (List key benefits of recommended schemes)
        
        ## 📋 Required Documents
        (List documents typically needed)
        
        ## 📝 Application Steps
        (Provide clear, actionable steps)
        
        ## 📚 Sources Used
        (List which documents were referenced)
        
        Important Guidelines:
        - Be specific and actionable
        - Include actual scheme names from documents
        - Mention official government sources
        - Add state-specific information if available
        - Keep response professional but friendly
        
        ---
        ⚠️ **Responsible AI Disclaimer**: This recommendation is based on available scheme documents and should be verified through official government portals.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}\n\n---\n⚠️ This recommendation is based on available scheme documents and should be verified through official government portals."

def run_agentic_workflow(query: str, rag_pipeline, model, k: int = 5) -> Dict:
    """
    Execute the complete agentic workflow:
    User Query → ProfileAgent → RetrievalAgent → EligibilityAgent → ResponseAgent
    """
    
    # Step 1: ProfileAgent - Extract user details
    profile_agent = ProfileAgent(model)
    profile = profile_agent.extract_profile(query)
    
    # Step 2: RetrievalAgent - Get relevant schemes
    retrieval_agent = RetrievalAgent(rag_pipeline)
    schemes_info, all_chunks = retrieval_agent.retrieve(query, profile, k=k)
    
    if not schemes_info:
        return {
            "profile": profile,
            "response": "No relevant scheme information found. Please try a different query or check if PDF documents are loaded properly.",
            "schemes": []
        }
    
    # Step 3: EligibilityAgent - Analyze eligibility
    eligibility_agent = EligibilityAgent(model)
    eligibility_results = eligibility_agent.analyze_eligibility(profile, schemes_info)
    
    # Step 4: ResponseAgent - Generate final answer
    response_agent = ResponseAgent(model)
    final_response = response_agent.generate_response(query, profile, schemes_info, eligibility_results)
    
    return {
        "profile": profile,
        "schemes": list(schemes_info.keys()),
        "eligibility": eligibility_results,
        "response": final_response
    }
