import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Define expanded keywords and responses
keywords = {
    'cycle': 'A regular menstrual cycle typically lasts between 21 to 35 days. Tracking it is important for understanding your body.',
    'irregular': 'Irregular periods can happen for many reasons. If you experience irregularities, consider consulting with a healthcare provider.',
    'hygiene': 'During your period, it’s important to maintain hygiene. Changing pads/tampons regularly and keeping your private areas clean is essential.',
    'mood': 'It’s normal to feel emotional during your period due to hormonal changes. Would you like to talk more about it?',
    'health': 'Pay attention to your overall health by maintaining a healthy diet and staying hydrated, especially during your period.',
    'cramps': 'Menstrual cramps are common. Try using a heating pad, exercising lightly, or taking pain relievers if needed.',
    'flow': 'Menstrual flow can vary. A normal period lasts between 3 and 7 days, and the flow can be light, medium, or heavy.',
    'ovulation': 'Ovulation typically occurs in the middle of your cycle, around 14 days before your next period, and is the time when you can get pregnant.',
    'fertility': 'Your fertility window is the time when you’re most likely to conceive, which occurs around ovulation.',
    'tampons': 'Tampons are an option for period care. Make sure to change them regularly to avoid infections.',
    'pads': 'Pads are another option. Make sure to change them regularly to avoid leakage and maintain hygiene.',
    'menopause': 'Menopause marks the end of your menstrual cycles. It usually happens between 45-55 years of age.',
    'puberty': 'Puberty is the stage when your body undergoes physical changes, including the start of menstruation, which typically occurs between ages 9 and 16.',
    'endometriosis': 'Endometriosis is a condition where tissue similar to the uterine lining grows outside the uterus, causing pain and other symptoms.',
    'PCOS': 'PCOS (Polycystic Ovary Syndrome) is a condition where the ovaries produce an abnormal amount of male hormones, leading to irregular periods and other symptoms.',
    'spotting': 'Spotting refers to light bleeding between periods and can occur for various reasons, such as hormonal changes or ovulation.',
    'menstrual cup': 'A menstrual cup is a reusable product placed inside the vagina to collect menstrual flow. It’s an eco-friendly alternative to pads and tampons.',
    'hormones': 'Hormones regulate your menstrual cycle. Fluctuations can cause symptoms like mood swings, cramps, or changes in the flow.',
    'pregnancy': 'If you miss a period or experience other symptoms, you might want to consider a pregnancy test.',
    'stress': 'Stress can affect your menstrual cycle. It can cause delays or make your period heavier or lighter than usual.',
    'exercise': 'Light exercise, like walking or yoga, can help ease menstrual cramps and improve mood during your period.',
    'hydration': 'Staying hydrated is especially important during your period to avoid fatigue and headaches.',
    'diet': 'Eating a balanced diet can help regulate your menstrual cycle. Iron-rich foods are important if your period causes heavy bleeding.',
    'bloating': 'Bloating is a common symptom of menstruation due to hormonal changes. Try to reduce salt intake and drink plenty of water.',
    'acne': 'Hormonal changes during your period can cause acne. Keeping your skin clean and hydrated may help.',
    'weight gain': 'Some people experience temporary weight gain during their period due to water retention.',
    'fatigue': 'Fatigue is common during menstruation, especially if you have heavy flow. Make sure to get enough sleep and take breaks as needed.',
    'anemia': 'Heavy menstrual bleeding can lead to anemia, which is a lack of red blood cells. Talk to your doctor if you feel tired or weak.',
    'discharge': 'It’s normal to have vaginal discharge between periods. It helps keep the vagina clean and healthy.',
    'vaginitis': 'Vaginitis is an inflammation of the vagina that can cause itching, discharge, or odor. It’s important to keep the area clean and see a doctor if symptoms persist.',
    'UTI': 'Urinary tract infections (UTIs) can occur more often during menstruation. Drink plenty of water and practice good hygiene.',
    'temperature': 'Your basal body temperature (BBT) may rise slightly after ovulation and drop before your period. Tracking BBT can help with cycle monitoring.',
    'tracking': 'There are many apps available to help track your period, cycle, and symptoms. These can help you understand your body better.',
    'irregularities': 'Irregular periods can be caused by stress, diet, or health conditions. If you notice significant changes, consult a healthcare professional.',
    'consult': 'If you have concerns about your cycle or symptoms, it’s always a good idea to consult with a healthcare provider.',
    'sensitive': 'Many people experience more sensitivity in their breasts and abdomen during their period due to hormonal changes.',
    'diarrhea': 'Some people may experience diarrhea or constipation around their period. Hormonal changes can affect your digestive system.',
    'tampon rash': 'If you experience irritation or rash from tampons, consider switching to pads or a menstrual cup and consult a doctor.',
    'vaginal health': 'Good vaginal health includes regular hygiene practices, wearing breathable underwear, and using products free from harmful chemicals.',
    'water retention': 'Water retention is common during periods and can cause swelling in the hands, feet, and abdomen.',
    'cycle length': 'Your menstrual cycle length can vary from month to month. It’s normal to have some variation, but drastic changes should be checked.',
    'late period': 'A late period can be caused by stress, diet, or hormonal changes. If it’s more than a few days late, consider taking a pregnancy test.',
    'early period': 'An early period can also be caused by hormonal changes, stress, or illness. Track your cycle to see if it becomes regular.',
    'post-period': 'After your period ends, you may still experience mild cramping or spotting. This is normal as your body adjusts.',
    'emotions': 'Hormonal fluctuations during your period can cause mood swings, irritability, or sadness. Take it easy and talk to someone if you need support.',
    'infections': 'Period-related infections, like yeast infections or bacterial vaginosis, can happen due to improper hygiene. See a doctor if you experience discomfort.',
    'fluid intake': 'Make sure to drink enough fluids to stay hydrated, especially during your period. Water, herbal teas, and clear broths are great options.',
    'calcium': 'Eating foods high in calcium can help reduce menstrual cramps and other symptoms of PMS.',
    'PMS': 'Premenstrual Syndrome (PMS) includes symptoms like irritability, cramps, and fatigue that occur before your period.',
    'clots': 'Blood clots during your period can be normal, especially if your flow is heavy. However, if you notice large clots or excessive bleeding, consult a doctor.',
    'breast tenderness': 'Breast tenderness is a common symptom of PMS and menstruation, caused by hormonal changes.',
    'exhaustion': 'Feeling really tired? It’s normal to feel exhausted during your period. Rest and hydration are key!',
    'irritable': 'Hormonal shifts can make us feel irritable. Take a deep breath and give yourself some grace during this time.',
    'fatigued': 'Fatigue is common during your period. Make sure to rest and stay hydrated!',
    'mood swings': 'Mood swings happen to a lot of us during our period. Want to talk about how you’re feeling?',
    'headache': 'Headaches can happen due to hormonal changes. A little rest and water can often help!',
    'cramps': 'Cramps can be uncomfortable, but there are ways to ease the pain—heat pads and light exercise can help.',
    'sore': 'Feeling sore? It could be from cramps or fatigue. Take it easy and try some gentle stretching.',
    'dizzy': 'Feeling dizzy is something some people experience during menstruation. If it persists, try sitting down, breathing deeply, and drinking water.',
    'low energy': 'Low energy is totally normal during your period. Make sure to take small breaks and listen to your body.',
    'fatigue': "Fatigue is common. If you're feeling drained, it's important to rest and recharge.",
    'nausea': 'Feeling a bit nauseous? That can happen, especially around your period. Ginger tea or some light snacks can help.',
    'bloating': 'Bloating is super common. Try drinking water and avoid salty foods to feel more comfortable.',
    'sensitive': 'You might feel extra sensitive right now—it’s all part of the hormonal changes happening in your body.',
    'down': 'It’s okay to feel down sometimes. Would you like to talk or take a break?',
    'emotional': 'It’s natural to feel emotional, especially with the hormonal changes. What are you feeling right now?',
    'stress': 'Stress can affect your body. Take a deep breath and try to relax—maybe even some light stretching will help!',
    'angry': 'It’s okay to feel angry, sometimes our hormones just make us feel more intense emotions. Want to talk about what’s bothering you?',
    'frustrated': 'Periods can be frustrating sometimes, especially with all the physical changes. Let me know if you need help or just need to vent.',
    'comfort': 'Comfort is so important. Whether it’s a warm blanket or some quiet time, make sure you give yourself some care.',
    'restless': 'If you’re feeling restless, it could be from a mix of hormonal changes and fatigue. A little rest might help.',
    'sad': 'It’s totally normal to feel a little sad during this time. Let yourself feel, and know you can talk to someone if you need support.',
    'relaxed': 'It’s great that you’re feeling relaxed! Try to hold onto that calm as you move through your day.',
    'cravings': 'Food cravings are a thing! Chocolate, salty snacks—your body’s asking for what it needs. Enjoy in moderation!',
    'hormones': 'Hormonal changes are the main cause of mood swings and physical symptoms. Your body is just going through some natural shifts.',
    'discomfort': 'Feeling discomfort can be a part of your period. If the pain gets too much, let me know if you need tips to ease it.',
    'tension': 'Tension can build up in your muscles around this time. A nice warm bath or a gentle massage could help ease it.',
    'headache relief': 'A headache can be tough, but staying hydrated and resting in a quiet space might give you some relief.',
    'period': 'Your period is a natural process where the body sheds the uterine lining. If you have any concerns, feel free to ask.',
    'exhausted': 'Exhaustion is something many people feel during their period. Make sure to take naps and rest as much as you need.',
    'moody': 'Feeling moody is completely normal when your hormones are shifting. How can I help make your day a little better?',
    'pain relief': 'For cramps, using a heating pad, staying hydrated, and light stretching can all help ease the discomfort.',
    'heavy': 'A heavy flow can be tough. Keep hydrated and take breaks when you need to.',
    'tired': 'Feeling tired is a common symptom. Your body needs more rest and hydration during this time, so be kind to yourself.',
    'calm': 'Trying to stay calm can be tricky with all the emotional changes. Deep breaths and a relaxing activity might help you feel better.',
    'peaceful': 'A peaceful environment can help you unwind and feel better. Would you like some suggestions on how to relax?',
    'guilty': 'It’s okay to feel guilty sometimes, but remember you’re doing your best. Take care of yourself!',
    'faint': 'Feeling faint? It might be due to low blood sugar or dehydration. Take a seat and drink some water.',
    'hopeless': 'If you’re feeling hopeless, it’s okay to reach out. You’re not alone, and your feelings are valid.',
    'apprehensive': 'Feeling a little apprehensive? It’s common to have worries during this time. What’s on your mind?',
    'grumpy': 'We all get a little grumpy sometimes, especially with the hormone changes. Want to talk about it or do something to relax?',
    'overwhelmed': 'If you’re feeling overwhelmed, try to break things down into smaller tasks. Taking a little break can help too.',
    'burnout': 'Burnout is tough. Make sure to rest and do something calming. You deserve it!',
    'reluctant': 'Feeling reluctant to do something? It’s okay—maybe taking it slow will help you feel more motivated.',
    'unmotivated': 'Feeling unmotivated is okay, especially with everything your body is going through right now. Take a break if you need to.',
    'gassy': 'Gas can happen, especially around your period. Try some light exercise or herbal tea to help.',
    'irritation': 'Irritation can happen when your hormones are fluctuating. Try to take a deep breath or talk to someone about it.',
    'low mood': 'Low mood can happen during menstruation. Taking it easy and checking in with yourself may help.',
    'overheated': 'Feeling overheated? Cooling off with water or staying in a cool environment can help.',
    'weary': 'Feeling weary during your period is normal. Listen to your body and rest when needed.',
    'uncomfortable': 'Uncomfortable? Try adjusting your position, using a heating pad, or lying down to ease any discomfort.',
    'tension headache': 'Tension headaches can happen. A little rest, stretching, or a cool compress might help.',
    'rest': 'It’s important to rest when you need to, especially when your body is going through a lot of changes. Take it easy!',
    'emotional outbursts': 'Emotional outbursts can happen due to hormonal changes. If you need to talk, I’m here for you.',
    'tired eyes': 'Tired eyes can happen when you’re feeling exhausted. Try some gentle eye massages or a little nap!',
    'burning': 'Burning sensations could indicate irritation. Take a break and consult a doctor if it doesn’t get better.',
    'tired legs': 'Tired legs are common with menstruation. A little stretching or an Epsom salt bath might help relax them.',
    'weakness': 'Weakness can occur, especially with heavy flow. Rest, stay hydrated, and eat iron-rich foods to help your energy levels.'
}

# Function to identify keywords in the input
def identify_keywords(user_input):
    # Tokenize the user input
    tokens = word_tokenize(user_input.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Search for keywords in the filtered input
    for word in filtered_tokens:
        if word in keywords:
            return keywords[word]
    return "I'm here to help! Could you tell me more about your concern?"

# Streamlit app UI
st.title("PeriodPal")

st.write("""
Hello! I'm your menstrual health confidante. You can talk to me about anything related to your cycle, hygiene, emotions, or anything else you'd like to share.
""")

# Text input for the user to chat with the bot
user_input = st.text_input("How can I help you today?")

if user_input:
    response = identify_keywords(user_input)
    st.write(response)

# Cycle tracking feature
st.write("## Track Your Cycle")

start_date = st.date_input("Start Date of Last Period")
end_date = st.date_input("End Date of Last Period")

if st.button("Calculate Cycle Length"):
    cycle_length = (end_date - start_date).days
    st.write(f"Your last cycle was {cycle_length} days long.")

    if cycle_length < 3 or cycle_length > 6:
        st.warning(
            "It seems your cycle length is a bit irregular. Consider consulting a healthcare provider if this continues.")

# Encourage emotional expression
st.write("## How Are You Feeling Today?")
mood = st.selectbox("Select your mood:", [
                    "Happy", "Anxious", "Irritated", "Sad", "Confused", "Other"])
if st.button("Share"):
    st.write(
        f"It's perfectly normal to feel {mood}. Remember, it's okay to express your emotions. Journaling or talking to someone you trust can help.")

st.write("Thank you for chatting! Remember, you're not alone on this journey.")

data = {
    'Symptoms': [
        'Cramps, Mood Swings, Fatigue',
        'Bloating, Headache, Acne',
        'Back Pain, Stress, Fatigue',
        'Nausea, Dizziness, Irritability',
        'Tender Breasts, Constipation, Mood Swings',
        'Sleep Disturbances, Stress, Anxiety',
        'Headache, Nausea, Bloating',
        'Fatigue, Joint Pain, Back Pain',
        'Mood Swings, Irritability, Stress',
        'Acne, Bloating, Food Cravings',
        'Headache, Fatigue, Dizziness',
        'Back Pain, Insomnia, Stress',
        'Fatigue, Low Appetite, Weakness',
        'Cramps, Bloating, Nausea',
        'Tender Breasts, Cravings, Mood Swings',
        'Acne, Hormonal Imbalance, Bloating',
        'Nausea, Fatigue, Hot Flashes',
        'Anxiety, Irritability, Weakness',
        'Stress, Insomnia, Headache',
        'Fatigue, Depression, Stress',
        'Back Pain, Headache, Anxiety',
        'Bloating, Constipation, Fatigue',
        'Mood Swings, Anxiety, Stress',
        'Cramps, Stress, Weakness',
        'Acne, Anxiety, Hormonal Imbalance',
        'Tender Breasts, Nausea, Cravings',
        'Dizziness, Weakness, Anxiety'
    ],
    'Category': [
        'Hormonal Imbalance', 'Acne & Digestion', 'Stress-related', 'Nausea/Vertigo', 
        'Hormonal Imbalance', 'Mental Health', 'Acne & Digestion', 'Stress-related', 
        'Mental Health', 'Acne & Digestion', 'Nausea/Vertigo', 'Stress-related', 
        'Fatigue & Weakness', 'Hormonal Imbalance', 'Hormonal Imbalance', 'Acne & Digestion', 
        'Hormonal Imbalance', 'Mental Health', 'Mental Health', 'Stress-related', 'Mental Health', 
        'Acne & Digestion', 'Mental Health', 'Fatigue & Weakness', 'Hormonal Imbalance', 
        'Acne & Digestion', 'Mental Health'
    ]
}

# Convert to DataFrame for AI model
df = pd.DataFrame(data)

# Use CountVectorizer to convert symptoms into a numerical format
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Symptoms'])

# Encode labels (categories)
encoder = LabelEncoder()
y = encoder.fit_transform(df['Category'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the classifier (Random Forest in this case)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit interface for symptom selection
st.title("Symptom Tracker Quiz")

# List of symptoms to select from (both for AI and non-AI advice)
symptom_options = [
    'Cramps', 'Mood Swings', 'Fatigue', 'Bloating', 'Headache', 'Acne', 
    'Back Pain', 'Nausea', 'Dizziness', 'Tender Breasts', 'Constipation', 
    'Diarrhea', 'Appetite Changes', 'Sleep Disturbances', 'Urinary Urgency', 
    'Hot Flashes', 'Stress', 'Irritability', 'Weakness', 'Chills', 
    'Joint Pain', 'Numbness', 'Skin Rashes', 'Vaginal Discharge', 'Cravings', 
    'Anxiety', 'Depression', 'Feeling Overwhelmed', 'Breast Tenderness'
]

# Collecting symptoms from the user using checkboxes
st.header("Please select your current symptoms:")
selected_symptoms = []

def give_advice(symptoms):
    advice = []

    # Providing advice for common symptoms
    if 'Cramps' in symptoms:
        advice.append("Consider using a heating pad to relieve cramps and take over-the-counter pain relievers if needed. Stretching exercises or light yoga may help.")
    
    if 'Mood Swings' in symptoms:
        advice.append("Try practicing mindfulness or meditation to help manage mood swings. Gentle exercise or talking to someone you trust can also help.")
    
    if 'Fatigue' in symptoms:
        advice.append("Ensure you're getting enough rest and staying hydrated. Healthy snacks with a mix of protein and carbs can give you a boost of energy.")
    
    if 'Bloating' in symptoms:
        advice.append("Try avoiding salty foods and eating smaller meals throughout the day. Staying hydrated and doing light exercises may also help with bloating.")
    
    if 'Headache' in symptoms:
        advice.append("For headaches, try a cool compress, rest in a dark room, and drink plenty of water. Over-the-counter pain relievers may help.")
    
    if 'Acne' in symptoms:
        advice.append("For acne, keep your skin clean and hydrated. You might also want to avoid touching your face too much and use non-comedogenic products.")
    
    if 'Back Pain' in symptoms:
        advice.append("Back pain can be relieved by using a heating pad or gentle stretching. Consider a relaxing yoga session or getting a massage.")
    
    if 'Nausea' in symptoms:
        advice.append("Try sipping ginger tea or staying hydrated. If nausea persists, it might be helpful to avoid heavy foods and opt for light snacks.")
    
    if 'Dizziness' in symptoms:
        advice.append("Make sure to stay hydrated and rest. If dizziness continues, it's best to consult a healthcare provider.")
    
    if 'Tender Breasts' in symptoms:
        advice.append("Breast tenderness can be eased by wearing a supportive bra. Try avoiding tight clothing and caffeine if it helps reduce discomfort.")
    
    if 'Constipation' in symptoms:
        advice.append("Eating fiber-rich foods, staying hydrated, and doing some gentle exercise can help ease constipation.")
    
    if 'Diarrhea' in symptoms:
        advice.append("For diarrhea, drink plenty of fluids to stay hydrated. Avoid dairy and greasy foods, and consider eating bland foods like rice or bananas.")
    
    if 'Appetite Changes' in symptoms:
        advice.append("It's normal to experience appetite changes. Try eating smaller, balanced meals throughout the day to help manage hunger.")
    
    if 'Sleep Disturbances' in symptoms:
        advice.append("Try to create a relaxing bedtime routine, avoid screen time before bed, and ensure your bedroom is cool and dark to improve sleep quality.")
    
    if 'Urinary Urgency' in symptoms:
        advice.append("Stay hydrated and avoid caffeine. If symptoms persist, it might be helpful to see a healthcare provider.")
    
    if 'Hot Flashes' in symptoms:
        advice.append("Staying cool with light clothing or fans and staying hydrated may help with hot flashes.")
    
    if 'Stress' in symptoms:
        advice.append("Take time to practice relaxation techniques like deep breathing, meditation, or even journaling to relieve stress.")
    
    if 'Irritability' in symptoms:
        advice.append("Taking regular breaks and practicing mindfulness can help manage irritability. Don't hesitate to talk to a friend or family member.")
    
    if 'Weakness' in symptoms:
        advice.append("Weakness can be related to fatigue or low iron. Resting, staying hydrated, and having a balanced diet might help.")
    
    if 'Chills' in symptoms:
        advice.append("Wear warm clothing and stay cozy. If chills persist, it might be a sign of something more, so consider seeing a doctor.")
    
    if 'Joint Pain' in symptoms:
        advice.append("Gentle exercises or stretching can help with joint pain. Over-the-counter pain relievers or warm baths may also provide relief.")
    
    if 'Numbness' in symptoms:
        advice.append("If you experience numbness frequently, it may be worth consulting a healthcare provider to rule out any underlying issues.")
    
    if 'Skin Rashes' in symptoms:
        advice.append("For skin rashes, consider using soothing creams or lotions. If they persist or worsen, it's best to consult a doctor.")
    
    if 'Vaginal Discharge' in symptoms:
        advice.append("It's normal to have vaginal discharge, but if you notice unusual color or odor, it’s important to consult with a healthcare provider.")
    
    if 'Cravings' in symptoms:
        advice.append("Cravings are common during certain times of the month. Try to balance your cravings with healthier alternatives like fruit or nuts.")
    
    if 'Anxiety' in symptoms:
        advice.append("Take deep breaths and focus on grounding exercises. It might also help to talk to someone you trust or take a break from stressors.")
    
    if 'Depression' in symptoms:
        advice.append("If you're feeling down for a prolonged period, don't hesitate to reach out for support from friends, family, or a healthcare professional.")
    
    if 'Feeling Overwhelmed' in symptoms:
        advice.append("Taking one task at a time and breaking big tasks into smaller ones can help. It’s okay to ask for help if you need it.")
    
    if 'Breast Tenderness' in symptoms:
        advice.append("Wear a comfortable and supportive bra. Avoid tight clothing and caffeine if they make the tenderness worse.")

    if not symptoms:
        advice.append("You're feeling great today! Keep taking care of yourself.")

    return advice

for symptom in symptom_options:
    if st.checkbox(symptom):
        selected_symptoms.append(symptom)

# When the user submits the symptoms
if st.button("Get Advice"):

    if selected_symptoms:
        st.subheader("Advice based on your symptoms:")
        advice = give_advice(selected_symptoms)
        for item in advice:
            st.write(f"- {item}")

        # AI-based Advice
        st.subheader("AI-Based Advice:")
        
        # Convert selected symptoms into the same format as the training data
        symptom_input = ', '.join(selected_symptoms)
        
        # Vectorize the symptoms input for AI model
        input_vector = vectorizer.transform([symptom_input])
        
        # Predict category using the trained model
        predicted_category = model.predict(input_vector)
        category = encoder.inverse_transform(predicted_category)[0]

        # Provide AI-generated advice based on predicted category
        if category == 'Hormonal Imbalance':
            st.write("AI-based advice: Consider balancing your hormones with a healthy diet, exercise, and stress management techniques. Consult a healthcare provider if symptoms persist.")
        elif category == 'Acne & Digestion':
            st.write("AI-based advice: Maintain a skin-care routine, stay hydrated, and consider dietary changes like reducing processed foods. Also, try adding fiber to your diet to improve digestion.")
        elif category == 'Stress-related':
            st.write("AI-based advice: Practice mindfulness, yoga, or deep breathing exercises. Try to manage stress by prioritizing rest and relaxation. Speaking to a therapist can also be helpful.")
        elif category == 'Nausea/Vertigo':
            st.write("AI-based advice: Ginger tea can help with nausea. Stay hydrated and avoid sudden head movements. If dizziness persists, consult a doctor.")
        elif category == 'Mental Health':
            st.write("AI-based advice: Try relaxation techniques such as meditation or talking to someone you trust. If you're feeling overwhelmed, consider reaching out to a mental health professional.")
    else:
        st.write("Please select at least one symptom.")