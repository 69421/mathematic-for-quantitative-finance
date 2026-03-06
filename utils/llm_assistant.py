from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

CFA_SYSTEM_PROMPT = """Tu es un assistant pédagogique expert en finance, spécialisé dans la préparation au CFA Level I.
Tu aides les étudiants francophones à comprendre les concepts financiers, à résoudre des exercices et à réviser efficacement.
Réponds toujours en français, de façon claire, structurée et pédagogique.
Utilise des exemples concrets et des analogies quand c'est pertinent."""


def get_llm(api_key: str, temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini", temperature=temperature)


def explain_concept(concept: str, context: str, api_key: str) -> str:
    llm = get_llm(api_key)
    messages = [
        SystemMessage(content=CFA_SYSTEM_PROMPT),
        HumanMessage(content=f"""Explique le concept suivant en te basant sur le contexte fourni:

Concept: {concept}

Contexte extrait du livre CFA:
{context}

Fournis:
1. Une explication claire et détaillée
2. Un résumé simplifié (2-3 phrases)
3. Un exemple illustratif concret
4. Les points clés à retenir""")
    ]
    response = llm.invoke(messages)
    return response.content


def translate_concept(term: str, context: str, api_key: str) -> str:
    llm = get_llm(api_key)
    messages = [
        SystemMessage(content=CFA_SYSTEM_PROMPT),
        HumanMessage(content=f"""Traduis et explique le terme financier suivant de l'anglais vers le français:

Terme: {term}

Contexte (si disponible):
{context}

Fournis:
1. La traduction française officielle
2. Une définition claire en français
3. L'usage dans le contexte CFA
4. Des termes synonymes ou liés""")
    ]
    response = llm.invoke(messages)
    return response.content


def solve_exercise(exercise: str, context: str, api_key: str) -> str:
    llm = get_llm(api_key, temperature=0.1)
    messages = [
        SystemMessage(content=CFA_SYSTEM_PROMPT),
        HumanMessage(content=f"""Résous l'exercice suivant étape par étape:

Exercice:
{exercise}

Contexte du programme CFA (si pertinent):
{context}

Fournis:
1. L'identification du type de problème et des concepts impliqués
2. La résolution étape par étape avec les formules utilisées
3. La réponse finale clairement indiquée
4. Les erreurs courantes à éviter pour ce type d'exercice
5. Les formules clés à mémoriser""")
    ]
    response = llm.invoke(messages)
    return response.content


def generate_revision_sheet(chapter_content: str, chapter_name: str, api_key: str) -> str:
    llm = get_llm(api_key, temperature=0.2)
    messages = [
        SystemMessage(content=CFA_SYSTEM_PROMPT),
        HumanMessage(content=f"""Génère une fiche de révision complète pour le chapitre suivant du CFA Level I:

Chapitre: {chapter_name}

Contenu:
{chapter_content[:3000]}

La fiche doit contenir:
1. 📌 Résumé synthétique (5-7 points clés)
2. 📐 Formules importantes (avec explications des variables)
3. 🔑 Concepts fondamentaux à maîtriser
4. ⚠️ Pièges et erreurs courantes à l'examen
5. 💡 Astuces mnémotechniques""")
    ]
    response = llm.invoke(messages)
    return response.content


def generate_flashcards(content: str, topic: str, num_cards: int, api_key: str) -> list[dict]:
    llm = get_llm(api_key, temperature=0.4)
    messages = [
        SystemMessage(content=CFA_SYSTEM_PROMPT),
        HumanMessage(content=f"""Génère exactement {num_cards} flashcards pour réviser le sujet suivant du CFA Level I:

Sujet: {topic}

Contenu de référence:
{content[:2000]}

Retourne les flashcards au format suivant (une par ligne, séparées par '|||'):
Question ||| Réponse

Assure-toi que:
- Les questions sont précises et testent la compréhension réelle
- Les réponses sont concises mais complètes
- Les flashcards couvrent différents aspects du sujet
- Le contenu est pertinent pour l'examen CFA Level I""")
    ]
    response = llm.invoke(messages)
    cards = []
    for line in response.content.strip().split('\n'):
        if '|||' in line:
            parts = line.split('|||', 1)
            if len(parts) == 2:
                question = parts[0].strip().lstrip('0123456789.-) ')
                answer = parts[1].strip()
                if question and answer:
                    cards.append({"question": question, "answer": answer, "topic": topic})
    return cards


def generate_exam_questions(topics: list[str], num_questions: int, api_key: str) -> list[dict]:
    llm = get_llm(api_key, temperature=0.5)
    topics_str = ", ".join(topics) if topics else "tous les sujets du CFA Level I"
    messages = [
        SystemMessage(content=CFA_SYSTEM_PROMPT),
        HumanMessage(content=f"""Génère exactement {num_questions} questions d'examen à choix multiple (QCM) sur les sujets suivants du CFA Level I: {topics_str}

Pour chaque question, utilise exactement ce format:
QUESTION: [La question]
A) [Option A]
B) [Option B]
C) [Option C]
REPONSE: [A, B ou C]
EXPLICATION: [Explication de la bonne réponse]
---

Assure-toi que:
- Les questions sont de niveau CFA Level I
- Les trois options sont plausibles
- Les questions couvrent calculs, concepts et interprétations
- Les explications sont pédagogiques""")
    ]
    response = llm.invoke(messages)
    questions = []
    blocks = response.content.strip().split('---')
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        try:
            lines = block.split('\n')
            question_text = ""
            options = {}
            correct = ""
            explanation = ""
            for line in lines:
                line = line.strip()
                if line.startswith('QUESTION:'):
                    question_text = line.replace('QUESTION:', '').strip()
                elif line.startswith('A)'):
                    options['A'] = line[2:].strip()
                elif line.startswith('B)'):
                    options['B'] = line[2:].strip()
                elif line.startswith('C)'):
                    options['C'] = line[2:].strip()
                elif line.startswith('REPONSE:'):
                    correct = line.replace('REPONSE:', '').strip()
                elif line.startswith('EXPLICATION:'):
                    explanation = line.replace('EXPLICATION:', '').strip()
            if question_text and len(options) == 3 and correct and explanation:
                questions.append({
                    "question": question_text,
                    "options": options,
                    "correct": correct,
                    "explanation": explanation
                })
        except Exception:
            continue
    return questions
