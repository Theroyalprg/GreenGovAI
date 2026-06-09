# 🌿 GreenGov AI

### Discovering Clean Energy, Water Conservation & Sustainable Agriculture Subsidies using RAG and Agentic AI



## 🎯 Overview

GreenGov AI is an intelligent assistant that helps farmers, entrepreneurs, and citizens discover government subsidies and schemes for:
- 🌞 **Clean Energy** (Solar pumps, renewable energy)
- 💧 **Water Conservation** (Drip irrigation, micro-irrigation)
- 🌾 **Sustainable Agriculture** (Soil health, crop insurance)

The system uses **RAG (Retrieval Augmented Generation)** with **Agentic AI** to provide personalized, context-aware recommendations based on official government documents.

## 🏗️ Architecture
User Query → ProfileAgent → RetrievalAgent → EligibilityAgent → ResponseAgent
↓ ↓ ↓ ↓ ↓
Question Extract Search FAISS Analyze Generate
User Info Vector DB Eligibility Final Answe
