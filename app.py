import os
import streamlit as st
from dotenv import load_dotenv
from utils.pdf_processor import load_pdf, split_documents
from utils.vector_store import build_vector_store, load_vector_store, search_documents
from utils.llm_assistant import (
    explain_concept, translate_concept, solve_exercise,
    generate_revision_sheet, generate_flashcards, generate_exam_questions
)
from utils.flashcards import save_flashcards, load_all_flashcards, get_topics, delete_flashcards_by_topic
from utils.exam_simulator import initialize_exam, get_remaining_time, submit_exam

load_dotenv()

st.set_page_config(
    page_title="Assistant CFA Level I",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    defaults = {
        "vector_store": None,
        "documents_loaded": False,
        "flashcards_session": [],
        "exam_state": None,
        "current_card_index": 0,
        "card_flipped": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY") or st.session_state.get("api_key")


def page_upload_documents():
    st.header("📤 Charger des documents CFA")
    st.markdown("Téléchargez vos livres du programme CFA au format PDF pour les analyser.")

    api_key = get_api_key()
    if not api_key:
        st.warning("⚠️ Veuillez configurer votre clé API OpenAI dans la barre latérale pour indexer les documents.")

    uploaded_files = st.file_uploader(
        "Sélectionner les fichiers PDF du programme CFA",
        type="pdf",
        accept_multiple_files=True,
        help="Vous pouvez télécharger plusieurs fichiers PDF à la fois."
    )

    if uploaded_files and api_key:
        if st.button("🔄 Analyser et indexer les documents", type="primary"):
            all_chunks = []
            progress = st.progress(0)
            status = st.empty()

            for i, file in enumerate(uploaded_files):
                status.text(f"Traitement de {file.name}...")
                docs = load_pdf(file)
                chunks = split_documents(docs)
                all_chunks.extend(chunks)
                progress.progress((i + 1) / len(uploaded_files))

            status.text("Construction de l'index vectoriel...")
            try:
                store = build_vector_store(all_chunks, api_key)
                st.session_state.vector_store = store
                st.session_state.documents_loaded = True
                st.success(f"✅ {len(uploaded_files)} document(s) indexé(s) avec succès ! ({len(all_chunks)} segments créés)")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'indexation : {str(e)}")

    elif uploaded_files and not api_key:
        st.info("Veuillez d'abord configurer votre clé API OpenAI.")

    if st.session_state.documents_loaded:
        st.success("📚 Documents déjà indexés et prêts à l'emploi.")
    elif api_key:
        store = load_vector_store(api_key)
        if store:
            st.session_state.vector_store = store
            st.session_state.documents_loaded = True
            st.info("📚 Index vectoriel existant chargé automatiquement.")


def page_explain_concept():
    st.header("💡 Explication de concepts")
    st.markdown("Posez une question ou sélectionnez un concept à expliquer.")

    api_key = get_api_key()
    if not api_key:
        st.error("❌ Clé API OpenAI manquante. Configurez-la dans la barre latérale.")
        return

    concept = st.text_area(
        "Concept ou question à expliquer",
        placeholder="Ex: Qu'est-ce que le ratio de Sharpe ? Comment calculer la duration modifiée ?",
        height=100
    )

    use_docs = st.checkbox("Utiliser les documents CFA chargés comme contexte", value=True)

    if st.button("💡 Expliquer", type="primary") and concept:
        context = ""
        if use_docs and st.session_state.vector_store:
            with st.spinner("Recherche dans les documents..."):
                docs = search_documents(st.session_state.vector_store, concept)
                context = "\n\n".join([d.page_content for d in docs])
        elif use_docs and not st.session_state.vector_store:
            st.info("ℹ️ Aucun document chargé. L'explication sera basée sur les connaissances générales.")

        with st.spinner("Génération de l'explication..."):
            try:
                result = explain_concept(concept, context, api_key)
                st.markdown("### 📖 Explication")
                st.markdown(result)
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")


def page_translation():
    st.header("🔄 Traduction & Glossaire")
    st.markdown("Traduisez et clarifiez les termes financiers anglais vers le français.")

    api_key = get_api_key()
    if not api_key:
        st.error("❌ Clé API OpenAI manquante. Configurez-la dans la barre latérale.")
        return

    term = st.text_input(
        "Terme financier anglais à traduire",
        placeholder="Ex: Duration, Convexity, Arbitrage, Bootstrapping..."
    )

    context = ""
    if st.session_state.vector_store and term:
        docs = search_documents(st.session_state.vector_store, term, k=2)
        context = "\n\n".join([d.page_content for d in docs])

    if st.button("🔄 Traduire & Expliquer", type="primary") and term:
        with st.spinner("Traduction en cours..."):
            try:
                result = translate_concept(term, context, api_key)
                st.markdown("### 📚 Résultat")
                st.markdown(result)
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")


def page_exercise_help():
    st.header("📝 Aide aux exercices")
    st.markdown("Soumettez un exercice pour obtenir une résolution détaillée étape par étape.")

    api_key = get_api_key()
    if not api_key:
        st.error("❌ Clé API OpenAI manquante. Configurez-la dans la barre latérale.")
        return

    exercise = st.text_area(
        "Énoncé de l'exercice",
        placeholder="Ex: Un investisseur achète une obligation avec un coupon annuel de 5%, une valeur nominale de 1000€ et une maturité de 3 ans. Si le taux de rendement requis est de 6%, calculez le prix de l'obligation.",
        height=200
    )

    if st.button("📐 Résoudre l'exercice", type="primary") and exercise:
        context = ""
        if st.session_state.vector_store:
            with st.spinner("Recherche de contexte pertinent..."):
                docs = search_documents(st.session_state.vector_store, exercise)
                context = "\n\n".join([d.page_content for d in docs])

        with st.spinner("Résolution en cours..."):
            try:
                result = solve_exercise(exercise, context, api_key)
                st.markdown("### 🔢 Résolution")
                st.markdown(result)
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")


def page_revision_sheets():
    st.header("📋 Fiches de révision")
    st.markdown("Générez des fiches de révision synthétiques pour chaque chapitre.")

    api_key = get_api_key()
    if not api_key:
        st.error("❌ Clé API OpenAI manquante. Configurez-la dans la barre latérale.")
        return

    cfa_topics = [
        "Ethical and Professional Standards",
        "Quantitative Methods",
        "Economics",
        "Financial Statement Analysis",
        "Corporate Issuers",
        "Equity Investments",
        "Fixed Income",
        "Derivatives",
        "Alternative Investments",
        "Portfolio Management",
    ]

    col1, col2 = st.columns([2, 1])
    with col1:
        chapter_name = st.selectbox("Sélectionnez un sujet CFA Level I", cfa_topics)
    with col2:
        custom_topic = st.text_input("Ou saisissez un sujet personnalisé")

    if custom_topic:
        chapter_name = custom_topic

    custom_content = st.text_area(
        "Contenu supplémentaire (optionnel)",
        placeholder="Collez ici du contenu spécifique sur lequel baser la fiche de révision...",
        height=100
    )

    if st.button("📋 Générer la fiche de révision", type="primary"):
        context = custom_content
        if st.session_state.vector_store and not custom_content:
            with st.spinner("Recherche de contenu pertinent..."):
                docs = search_documents(st.session_state.vector_store, chapter_name, k=6)
                context = "\n\n".join([d.page_content for d in docs])

        if not context:
            context = f"Chapitre CFA Level I: {chapter_name}"

        with st.spinner("Génération de la fiche de révision..."):
            try:
                result = generate_revision_sheet(context, chapter_name, api_key)
                st.markdown(f"### 📋 Fiche de révision : {chapter_name}")
                st.markdown(result)

                st.download_button(
                    label="⬇️ Télécharger la fiche (Markdown)",
                    data=result,
                    file_name=f"fiche_{chapter_name.replace(' ', '_')}.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")


def page_flashcards():
    st.header("🃏 Flashcards")

    api_key = get_api_key()

    tab1, tab2 = st.tabs(["➕ Créer des flashcards", "📖 Réviser"])

    with tab1:
        st.markdown("Générez des flashcards à partir d'un sujet ou de vos documents.")

        if not api_key:
            st.error("❌ Clé API OpenAI manquante. Configurez-la dans la barre latérale.")
        else:
            topic = st.text_input(
                "Sujet des flashcards",
                placeholder="Ex: Duration et convexité, Ratio financiers, Greeks des options..."
            )
            num_cards = st.slider("Nombre de flashcards à générer", 5, 20, 10)

            if st.button("🃏 Générer les flashcards", type="primary") and topic:
                context = ""
                if st.session_state.vector_store:
                    docs = search_documents(st.session_state.vector_store, topic, k=4)
                    context = "\n\n".join([d.page_content for d in docs])

                with st.spinner("Génération des flashcards..."):
                    try:
                        cards = generate_flashcards(context or topic, topic, num_cards, api_key)
                        save_flashcards(cards)
                        st.success(f"✅ {len(cards)} flashcards créées et sauvegardées pour le sujet : {topic}")

                        with st.expander("Aperçu des flashcards créées"):
                            for i, card in enumerate(cards, 1):
                                st.markdown(f"**{i}. Q:** {card['question']}")
                                st.markdown(f"   **R:** {card['answer']}")
                                st.divider()
                    except Exception as e:
                        st.error(f"❌ Erreur : {str(e)}")

    with tab2:
        all_cards = load_all_flashcards()

        if not all_cards:
            st.info("ℹ️ Aucune flashcard disponible. Créez-en d'abord dans l'onglet 'Créer des flashcards'.")
            return

        st.markdown(f"**{len(all_cards)} flashcards disponibles**")

        topics = get_topics()
        selected_topic = st.selectbox("Filtrer par sujet", ["Tous"] + topics)

        if selected_topic != "Tous":
            filtered_cards = [c for c in all_cards if c.get("topic") == selected_topic]
        else:
            filtered_cards = all_cards

        if not filtered_cards:
            st.warning("Aucune flashcard pour ce sujet.")
            return

        if "current_card_index" not in st.session_state or st.session_state.current_card_index >= len(filtered_cards):
            st.session_state.current_card_index = 0
        if "card_flipped" not in st.session_state:
            st.session_state.card_flipped = False

        idx = st.session_state.current_card_index
        card = filtered_cards[idx]

        st.markdown(f"**Carte {idx + 1} / {len(filtered_cards)}** — Sujet: {card.get('topic', 'Général')}")

        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            if not st.session_state.card_flipped:
                st.info(f"**❓ Question**\n\n{card['question']}")
                if st.button("🔄 Voir la réponse", use_container_width=True):
                    st.session_state.card_flipped = True
                    st.rerun()
            else:
                st.success(f"**✅ Réponse**\n\n{card['answer']}")
                if st.button("🔄 Voir la question", use_container_width=True):
                    st.session_state.card_flipped = False
                    st.rerun()

        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Précédent", use_container_width=True, disabled=(idx == 0)):
                st.session_state.current_card_index = max(0, idx - 1)
                st.session_state.card_flipped = False
                st.rerun()
        with col_next:
            if st.button("Suivant ➡️", use_container_width=True, disabled=(idx >= len(filtered_cards) - 1)):
                st.session_state.current_card_index = min(len(filtered_cards) - 1, idx + 1)
                st.session_state.card_flipped = False
                st.rerun()


def page_exam_simulator():
    st.header("🎯 Simulation d'examen")

    api_key = get_api_key()
    if not api_key:
        st.error("❌ Clé API OpenAI manquante. Configurez-la dans la barre latérale.")
        return

    if st.session_state.exam_state and not st.session_state.exam_state.get("completed"):
        exam = st.session_state.exam_state
        remaining = get_remaining_time(exam)
        questions = exam["questions"]

        minutes = remaining // 60
        seconds = remaining % 60

        if remaining == 0:
            results = submit_exam(exam)
            exam["completed"] = True
            exam["score"] = results
            st.rerun()

        col_timer, col_progress = st.columns([1, 3])
        with col_timer:
            if remaining < 300:
                st.error(f"⏱️ Temps restant: {minutes:02d}:{seconds:02d}")
            else:
                st.info(f"⏱️ Temps restant: {minutes:02d}:{seconds:02d}")
        with col_progress:
            answered = len(exam["answers"])
            st.progress(answered / len(questions), text=f"{answered}/{len(questions)} questions répondues")

        st.markdown("---")
        for i, q in enumerate(questions):
            st.markdown(f"**Question {i + 1}:** {q['question']}")
            options = [f"{k}) {v}" for k, v in q["options"].items()]
            current_answer = exam["answers"].get(i)

            current_idx = None
            if current_answer:
                for j, opt in enumerate(options):
                    if opt.startswith(current_answer):
                        current_idx = j
                        break

            answer = st.radio(
                f"q_{i}",
                options,
                index=current_idx,
                key=f"exam_q_{i}",
                label_visibility="collapsed"
            )
            if answer:
                exam["answers"][i] = answer[0]  # First character is A, B, or C
            st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Soumettre l'examen", type="primary", use_container_width=True):
                results = submit_exam(exam)
                exam["completed"] = True
                exam["score"] = results
                st.rerun()
        with col2:
            if st.button("❌ Abandonner l'examen", use_container_width=True):
                st.session_state.exam_state = None
                st.rerun()

    elif st.session_state.exam_state and st.session_state.exam_state.get("completed"):
        results = st.session_state.exam_state["score"]
        score_pct = results["score_pct"]

        st.markdown("## 📊 Résultats de l'examen")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{score_pct:.1f}%")
        with col2:
            st.metric("Correctes", f"{results['correct']}/{results['total']}")
        with col3:
            if score_pct >= 70:
                st.metric("Résultat", "✅ Réussi", delta="≥ 70%")
            else:
                st.metric("Résultat", "❌ Échec", delta="< 70%")

        st.markdown("---")
        st.markdown("### 📝 Analyse détaillée")

        for i, r in enumerate(results["results"]):
            icon = "✅" if r["is_correct"] else "❌"
            with st.expander(f"{icon} Question {i + 1}: {r['question'][:80]}..."):
                for key, val in r["options"].items():
                    if key == r["correct_answer"]:
                        st.markdown(f"**✅ {key}) {val}** ← Bonne réponse")
                    elif key == r["user_answer"]:
                        st.markdown(f"**❌ {key}) {val}** ← Votre réponse")
                    else:
                        st.markdown(f"{key}) {val}")
                st.info(f"💡 Explication: {r['explanation']}")

        if st.button("🔄 Nouveau test", type="primary"):
            st.session_state.exam_state = None
            st.rerun()

    else:
        st.markdown("Configurez votre simulation d'examen.")

        cfa_topics = [
            "Ethical and Professional Standards",
            "Quantitative Methods",
            "Economics",
            "Financial Statement Analysis",
            "Corporate Issuers",
            "Equity Investments",
            "Fixed Income",
            "Derivatives",
            "Alternative Investments",
            "Portfolio Management",
        ]

        selected_topics = st.multiselect(
            "Sujets à inclure",
            cfa_topics,
            default=cfa_topics[:3],
            help="Sélectionnez les matières à inclure dans l'examen"
        )

        col1, col2 = st.columns(2)
        with col1:
            num_questions = st.slider("Nombre de questions", 5, 30, 10)
        with col2:
            duration = st.slider("Durée (minutes)", 10, 120, 30)

        if st.button("🚀 Démarrer l'examen", type="primary", disabled=not selected_topics):
            with st.spinner("Génération des questions d'examen..."):
                try:
                    questions = generate_exam_questions(selected_topics, num_questions, api_key)
                    if not questions:
                        st.error("❌ Impossible de générer les questions. Réessayez.")
                        return
                    exam_state = initialize_exam(questions, duration)
                    st.session_state.exam_state = exam_state
                    st.success(f"✅ {len(questions)} questions générées !")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")


def main():
    init_session_state()

    with st.sidebar:
        st.title("📘 Assistant CFA Level I")
        st.markdown("*Votre compagnon intelligent pour la préparation au CFA*")
        st.divider()

        api_key_env = os.getenv("OPENAI_API_KEY")
        if not api_key_env:
            api_key_input = st.text_input(
                "🔑 Clé API OpenAI",
                type="password",
                placeholder="sk-...",
                help="Entrez votre clé API OpenAI pour utiliser l'assistant."
            )
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.success("✅ Clé API configurée")
        else:
            st.success("✅ Clé API chargée depuis l'environnement")

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "📤 Charger des documents",
                "💡 Expliquer un concept",
                "🔄 Traduction & Glossaire",
                "📝 Aide aux exercices",
                "📋 Fiches de révision",
                "🃏 Flashcards",
                "🎯 Simulation d'examen",
            ],
            label_visibility="collapsed"
        )

        st.divider()
        if st.session_state.documents_loaded:
            st.success("📚 Documents indexés")
        else:
            st.warning("📚 Aucun document chargé")

        st.caption("© 2024 – Application CFA Level I")

    if page == "📤 Charger des documents":
        page_upload_documents()
    elif page == "💡 Expliquer un concept":
        page_explain_concept()
    elif page == "🔄 Traduction & Glossaire":
        page_translation()
    elif page == "📝 Aide aux exercices":
        page_exercise_help()
    elif page == "📋 Fiches de révision":
        page_revision_sheets()
    elif page == "🃏 Flashcards":
        page_flashcards()
    elif page == "🎯 Simulation d'examen":
        page_exam_simulator()


if __name__ == "__main__":
    main()
