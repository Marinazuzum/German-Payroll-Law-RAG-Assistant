from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PromptStrategy(Enum):
    """Different prompt strategies for legal Q&A."""
    BASIC = "basic"
    STRUCTURED = "structured"
    LEGAL_EXPERT = "legal_expert"
    STEP_BY_STEP = "step_by_step"
    COMPARATIVE = "comparative"

class PromptBuilder:
    """Build prompts for different legal Q&A strategies."""
    
    def __init__(self):
        self.strategies = {
            PromptStrategy.BASIC: self._build_basic_prompt,
            PromptStrategy.STRUCTURED: self._build_structured_prompt,
            PromptStrategy.LEGAL_EXPERT: self._build_legal_expert_prompt,
            PromptStrategy.STEP_BY_STEP: self._build_step_by_step_prompt,
            PromptStrategy.COMPARATIVE: self._build_comparative_prompt,
        }
    
    def build_prompt(self, 
                    strategy: PromptStrategy, 
                    query: str, 
                    retrieved_documents: List[Dict[str, Any]],
                    include_sources: bool = True) -> str:
        """Build prompt based on strategy."""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return self.strategies[strategy](query, retrieved_documents, include_sources)
    
    def _format_documents(self, documents: List[Dict[str, Any]], include_metadata: bool = True) -> str:
        """Format retrieved documents for inclusion in prompt."""
        if not documents:
            return "Keine relevanten Dokumente gefunden."
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            content = doc['content'].strip()
            
            if include_metadata and 'metadata' in doc:
                metadata = doc['metadata']
                source = metadata.get('source', 'Unbekannt')
                file_name = metadata.get('file_name', 'Unbekannt')
                
                doc_text = f"**Dokument {i}** (Quelle: {file_name}):\n{content}"
            else:
                doc_text = f"**Dokument {i}**:\n{content}"
            
            formatted_docs.append(doc_text)
        
        return "\n\n".join(formatted_docs)
    
    def _build_basic_prompt(self, query: str, documents: List[Dict[str, Any]], include_sources: bool) -> str:
        """Basic prompt strategy."""
        context = self._format_documents(documents, include_metadata=include_sources)
        
        prompt = f"""Basierend auf den folgenden Dokumenten zum deutschen Lohnsteuerrecht, beantworte die Frage präzise und genau.

KONTEXT:
{context}

FRAGE: {query}

ANTWORT:"""
        
        return prompt
    
    def _build_structured_prompt(self, query: str, documents: List[Dict[str, Any]], include_sources: bool) -> str:
        """Structured prompt with clear sections."""
        context = self._format_documents(documents, include_metadata=include_sources)
        
        prompt = f"""Sie sind ein Experte für deutsches Lohnsteuerrecht. Analysieren Sie die bereitgestellten Dokumente und beantworten Sie die Frage strukturiert.

## VERFÜGBARE DOKUMENTE:
{context}

## BENUTZERANFRAGE:
{query}

## ANTWORTFORMAT:
Bitte strukturieren Sie Ihre Antwort wie folgt:

**Zusammenfassung:** 
[Kurze, direkte Antwort auf die Frage]

**Detaillierte Erklärung:**
[Ausführliche Erläuterung basierend auf den Dokumenten]

**Rechtliche Grundlagen:**
[Relevante Gesetze, Paragraphen oder Vorschriften]

**Praktische Hinweise:**
[Anwendbare Tipps oder wichtige Punkte für die Praxis]

## IHRE ANTWORT:"""
        
        return prompt
    
    def _build_legal_expert_prompt(self, query: str, documents: List[Dict[str, Any]], include_sources: bool) -> str:
        """Legal expert persona prompt."""
        context = self._format_documents(documents, include_metadata=include_sources)
        
        prompt = f"""Sie sind ein erfahrener Steuerberater und Experte für deutsches Lohnsteuerrecht mit über 20 Jahren Berufserfahrung. Ein Mandant stellt Ihnen eine Frage zum Lohnsteuerrecht.

VERFÜGBARE RECHTSDOKUMENTE:
{context}

MANDANTENANFRAGE:
{query}

Als erfahrener Steuerexperte analysieren Sie die Rechtslage sorgfältig und geben eine fundierte, praxisnahe Antwort. Berücksichtigen Sie dabei:

1. Die aktuellen gesetzlichen Bestimmungen
2. Mögliche Ausnahmen oder Sonderfälle
3. Praktische Auswirkungen für den Mandanten
4. Wichtige Fristen oder Verfahrensschritte

Formulieren Sie Ihre Antwort professionell, aber verständlich, und weisen Sie auf eventuelle Risiken oder Unsicherheiten hin.

IHRE EXPERTENMEINUNG:"""
        
        return prompt
    
    def _build_step_by_step_prompt(self, query: str, documents: List[Dict[str, Any]], include_sources: bool) -> str:
        """Step-by-step analysis prompt."""
        context = self._format_documents(documents, include_metadata=include_sources)
        
        prompt = f"""Analysieren Sie die folgende Frage zum deutschen Lohnsteuerrecht Schritt für Schritt.

VERFÜGBARE DOKUMENTE:
{context}

FRAGE: {query}

Gehen Sie dabei systematisch vor:

**Schritt 1: Problemidentifikation**
[Identifizieren Sie das Kernproblem der Frage]

**Schritt 2: Rechtliche Einordnung**
[Ordnen Sie die Frage in den rechtlichen Kontext ein]

**Schritt 3: Dokumentenanalyse**
[Analysieren Sie die relevanten Inhalte der Dokumente]

**Schritt 4: Rechtsanwendung**
[Wenden Sie die gefundenen Regelungen auf die Frage an]

**Schritt 5: Schlussfolgerung**
[Ziehen Sie eine begründete Schlussfolgerung]

**Schritt 6: Praktische Umsetzung**
[Geben Sie konkrete Handlungsempfehlungen]

IHRE SCHRITT-FÜR-SCHRITT-ANALYSE:"""
        
        return prompt
    
    def _build_comparative_prompt(self, query: str, documents: List[Dict[str, Any]], include_sources: bool) -> str:
        """Comparative analysis prompt."""
        context = self._format_documents(documents, include_metadata=include_sources)
        
        prompt = f"""Führen Sie eine vergleichende Analyse der verfügbaren Informationen zum deutschen Lohnsteuerrecht durch.

VERFÜGBARE QUELLEN:
{context}

ANALYSEANFRAGE: {query}

Führen Sie eine strukturierte Vergleichsanalyse durch:

**Übereinstimmende Punkte:**
[Welche Aspekte sind in allen relevanten Dokumenten konsistent?]

**Unterschiede oder Variationen:**
[Gibt es unterschiedliche Interpretationen oder Anwendungsfälle?]

**Vollständigkeit der Information:**
[Welche Aspekte der Frage können vollständig beantwortet werden?]
[Welche Bereiche benötigen zusätzliche Informationen?]

**Gewichtung der Quellen:**
[Bewerten Sie die Relevanz und Autorität der verschiedenen Dokumente]

**Synthese:**
[Erstellen Sie eine ausgewogene Antwort basierend auf der Vergleichsanalyse]

**Empfohlenes Vorgehen:**
[Schlagen Sie konkrete nächste Schritte vor]

IHRE VERGLEICHSANALYSE:"""
        
        return prompt
    
    def get_strategy_description(self, strategy: PromptStrategy) -> str:
        """Get description of a prompt strategy."""
        descriptions = {
            PromptStrategy.BASIC: "Einfache, direkte Antwort basierend auf den Dokumenten",
            PromptStrategy.STRUCTURED: "Strukturierte Antwort mit klaren Abschnitten",
            PromptStrategy.LEGAL_EXPERT: "Antwort aus der Perspektive eines erfahrenen Steuerberaters",
            PromptStrategy.STEP_BY_STEP: "Schritt-für-Schritt-Analyse der rechtlichen Frage",
            PromptStrategy.COMPARATIVE: "Vergleichende Analyse verschiedener Aspekte"
        }
        return descriptions.get(strategy, "Unbekannte Strategie")
    
    def get_all_strategies(self) -> List[PromptStrategy]:
        """Get list of all available strategies."""
        return list(PromptStrategy)

# Test functions
def test_prompt_strategies():
    """Test all prompt strategies with sample data."""
    builder = PromptBuilder()
    
    sample_query = "Wie werden Überstunden in Deutschland besteuert?"
    sample_documents = [
        {
            'content': "Überstunden sind grundsätzlich steuerpflichtiger Arbeitslohn. Sie unterliegen der Lohnsteuer und den Sozialversicherungsbeiträgen.",
            'metadata': {'source': '/path/to/doc1.pdf', 'file_name': 'lohnsteuer_grundlagen.pdf'},
            'score': 0.85
        },
        {
            'content': "Bei Überstunden ist zu beachten, dass diese zum normalen Steuersatz versteuert werden. Ausnahmen gibt es nur bei bestimmten Zuschlägen.",
            'metadata': {'source': '/path/to/doc2.pdf', 'file_name': 'steuer_sonderfaelle.pdf'},
            'score': 0.78
        }
    ]
    
    print("Testing all prompt strategies:\n")
    
    for strategy in builder.get_all_strategies():
        print(f"{'='*50}")
        print(f"STRATEGY: {strategy.value.upper()}")
        print(f"Description: {builder.get_strategy_description(strategy)}")
        print(f"{'='*50}")
        
        prompt = builder.build_prompt(strategy, sample_query, sample_documents)
        print(prompt)
        print(f"\n{'='*50}\n")

if __name__ == "__main__":
    test_prompt_strategies()
