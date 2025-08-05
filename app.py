import pandas as pd
import streamlit as st
import joblib
from datetime import datetime, date
import numpy as np

st.set_page_config(initial_sidebar_state='expanded',page_title='Prédicteur la churn - Youssouf',page_icon='🏦',layout='wide')

# source
@st.cache_resource
def charger_model():
    try:
        model = joblib.load('modelGBoost.pkl')
        features = joblib.load('features.pkl')
        return model,features
    except FileNotFoundError as e:
        st.error(f'Erreur: Fichier manquant - {e}')
        st.stop()
    except Exception as e:
        st.error(f'Erreur lors de chargement :{e}')
        st.stop()
        
model,features = charger_model()

# en tete
st.markdown('''
            <style>
            .main-header{
               background: linear-gradient(135deg, #91BDF2 0%, #91BDF2 100%);
               padding:2.2rem;
               border-radius:50px;
               margin-bottom:2rem;
               text-align:center;
               border-shadow: 0 20px 50px rgba(0,0,0,0.1);
                }
            </style>
            ''',unsafe_allow_html=True)
st.markdown('''
            <div class='main-header'>
            <h1>🏦 Prédicteur si un client va quitte l'entreprise</h1>
            <p style='font-size:20px;'>Développé par - <strong>Youssouf</strong> Assistant Intelligent</p>
            </div>
            
            ''',unsafe_allow_html=True)
# sidbar
st.markdown('''
            <style>
            .friendly-info {
                background: #e3f2fd;
                padding: 2rem;
                border-radius: 15px;
                border-left: 5px solid #2196F3;
                margin: 1.5rem 0;
            }
            .encouragement {
            background: linear-gradient(135deg, #fff3e0, #ffecb3);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border-left: 5px solid #ff9800;
        }
            </style>
            ''',unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## 🤖 À propos de votre assistant")
    st.markdown("""
    <div class="friendly-info">
        <h4>Comment je fonctionne ?</h4>
        <p>• J'utilise un modèle d'IA entraîné sur des milliers de cas</p>
        <p>• Ma précision est d'environ 98%</p>
        <p>• Je respecte votre vie privée</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 💡 Rappel important")
    st.markdown("""
    <div class="encouragement">
        <p><strong>Gardez en tête :</strong></p>
        <p>✨ Je suis un outil d'aide, pas un agent de la campagne de marketing</p>
    </div>
    """, unsafe_allow_html=True)

# formulaire
st.markdown('''
            <h2 style='color:#343a40;text-align:center;margin-bottom:25px'> 📋 Informations du clients</h2>
            
            ''',unsafe_allow_html=True)
# --- Formulaire client ---
with st.form(key='formulaire_client'):
    # 🧍 Informations personnelles
    st.header("🧍 Informations personnelles")
    nom_client = st.text_input(
        '📝 Votre nom complet du client', 
        placeholder="Ex: Youssouf Mohamed ",
        help="Saisissez le nom complet du client"
    ).strip()
    Age = st.slider('🎂 Âge du client', 18, 100, 1)
    Sexe = st.selectbox('🎓 Votre sexe', ['Femme', 'Homme'])
    Type_Abonnement = st.selectbox('💑 Votre type de abonnement', ['Basique', 'Stantard', 'prenium'])
    Duree_Contact = st.selectbox('📝Duree de contact', ['Annuel', 'Mensuel', 'Trimestriel'])

    # 🛍️ Dépenses
    st.header("🛍️ Dépenses produits")
    Anciennete = st.slider("⌛ Ancienneté", 1, 60, 1,help='veuillez entrer ancienneté')
    Utilisation_Service = st.slider("🛠️ Fréquence", 1, 40, 1,help='veuillez entrer fréquence d\'utilisation du service')
    Nombre_Appel = st.slider("📞  Nombre d\'appel", 0, 10, 1,help='veuillez entrer  Nombre d\'appel au service support')
    Delai_Paiement = st.slider("💳 Délai paiement", 0, 30, 1,help='veuillez entrer Délai de paiement moyen')
    Montant_Total = st.slider("💲 Montant Total", 100, 2000, 1,help='veuillez entrer  Montant Total')
    Derniere_Interaction = st.slider("📱 Dernière Interaction", 1, 30, 1,help='veuillez entrer  Dernière Interaction')
    st.markdown('---')
    col_center = st.columns([1,2,1])
    with col_center[1]:
        submit = st.form_submit_button(
            'Prédire la probabilité de souscription', 
            type="primary", 
            use_container_width=True
        )
if submit:
    if not nom_client:
        st.warning('Veuillez renseigner le nom complet du client !')
    else:
        donnees_client = {colonne: 0 for colonne in features}
       # donnee numerique
        donnees_client['Anciennete'] = Anciennete
        donnees_client['Utilisation_Service'] = Utilisation_Service
        donnees_client['Nombre_Appel'] = Nombre_Appel
        donnees_client['Delai_Paiement'] = Delai_Paiement
        donnees_client['Montant_Total'] = Montant_Total
        donnees_client['Derniere_Interaction'] = Derniere_Interaction
        # encodage
        Sexe = f'Sexe_{Sexe}'
        if Sexe in donnees_client:
            donnees_client[Sexe] = 1
            
        Type_Abonnement = f'Type_Abonnement_{Type_Abonnement}'
        if Type_Abonnement in donnees_client:
            donnees_client[Type_Abonnement] = 1
        Duree_Contact = f'Duree_Contact_{Duree_Contact}'
        if Duree_Contact in donnees_client:
            donnees_client[Duree_Contact] = 1
            
        # creation data
        nouvelle_donnee = pd.DataFrame([
            [donnees_client[col] for col in features]
            ],columns=features)    
        try:
            prediction = model.predict(nouvelle_donnee)[0]
            proba = model.predict_proba(nouvelle_donnee)[0][1]
            st.markdown('---')
            st.markdown(f"""
            <div class="friendly-info">
                <h2>Résultat de l\'analyse pour {nom_client}</h2>
            </div>
            """, unsafe_allow_html=True)
            if prediction == 0:
                st.success(f'✅ **Prédiction positive** :{nom_client} a de forte chances de reste l\'entreprise ! ')
                conseil = '**Recommadation** : Contactez ce client rapidement , il présente un profil trés  favorable'
            else:
                st.warning(f'❌ **Prédiction Négative** :{nom_client} a risque de quitter l\'entreprise ! ')
                conseil = '**Recommadation** : ce client ne néccesite un approche commercial adapté ou cible'
            col_prob1,col_prob2 = st.columns([1,2])
            with col_prob1:
                delta_val = float(round(proba - 0.5,2))
                st.metric(
                    label='🎯 Probabilité de accepte',
                    value =f'{proba:.1%}',
                    delta=delta_val
                    )
                st.caption('Différence par rapport à une moyenne de 50 %')
            with col_prob2:
                couleur_barre = "#28a745" if proba > 0.5 else "#dc3545"
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                    <h5 style="margin-bottom: 10px; color: #495057;">Niveau de confiance</h5>
                    <div style="background-color: #e9ecef; border-radius: 25px; height: 20px; overflow: hidden;">
                        <div style="width: {proba*100}%; height: 100%; background-color: {couleur_barre}; 
                                   border-radius: 25px; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.info(conseil)
        except Exception as e:
             st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
             st.info("💡 Veuillez vérifier que tous les champs sont correctement remplis.")     
         
# Message de conclusion plus chaleureux
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h4 style="color: #495057; margin-bottom: 1rem;">🏦' Votre Assistant Intelligente</h4>
    <p style="font-size: 1em; color: #6c757d; margin-bottom: 0.5rem;">
        Créé avec passion par <strong>Youssouf</strong> pour vous accompagner dans votre parcours santé
    </p>
    <p style="font-size: 0.9em; color: #6c757d; margin-bottom: 1rem;">
        Version 2024 - Mis à jour régulièrement pour votre bien-être
    </p>
    <div style="border-top: 1px solid #dee2e6; padding-top: 1rem;">
        <p style="font-size: 0.85em; color: #6c757d; font-style: italic;">
            ⚠️ Rappel important : Cet outil d'aide à la décision complète mais ne remplace jamais 
            l'expertise de votre agent
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

















    