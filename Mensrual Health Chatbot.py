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
    'period': 'Your period is a natural process where the body sheds the uterine lining. If you have any concerns, feel free to ask.',
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

    if cycle_length < 21 or cycle_length > 35:
        st.warning(
            "It seems your cycle length is a bit irregular. Consider consulting a healthcare provider if this continues.")

from textblob import TextBlob

# Analyze sentiment using TextBlob
@st.cache_resource
def analyze_sentiment(text):
    analysis = TextBlob(text).sentiment
    return analysis.polarity

# Streamlit UI
st.title("Emotion Journal")
st.write("Tell me how you're feeling, and I'll try to respond appropriately!")

# Input from the user
user_input = st.text_input("How are you feeling today?")

if user_input:
    # Get sentiment polarity
    polarity = analyze_sentiment(user_input)

    # Provide a response based on sentiment polarity
    if polarity > 0:
        st.write("🌟 I'm glad you're feeling good! Keep up the positive vibes!")
    elif polarity < 0:
        st.write("💔 I'm sorry you're feeling this way. Remember, it's okay to have tough days.")
    else:
        st.write("Thanks for sharing. I'm here to chat if you want to talk more!")

# Updated dataset with the new symptoms
data = {
    'Symptoms': [
        'Pain in Lower Abdomen, Urinary Pain, Nausea/Vertigo',
        'Headache, Breast Tenderness, Digestive Problems',
        'Diarrhea, Constipation, Feeling of being impure',
        'Sadness, Emotional lability, Anxiety',
        'Irritability/anger, Impulsiveness, Decreased appetite',
        'Increased appetite, Fatigue, Pain during sexual intercourse',
        'Headache, Fatigue, Anxiety',
        'Pain in Lower Abdomen, Irritability/anger, Anxiety',
        'Breast Tenderness, Insomnia, Hypersomnia',
        'Concentration impairment, Fatigue, Decreased sexual drive',
        'Urinary Pain, Nausea/Vertigo, Headache',
        'Digestive Problems, Constipation, Decreased appetite',
        'Sadness, Emotional lability, Insomnia',
        'Diarrhea, Discomfort due to vaginal bleeding, Pain during sexual intercourse',
        'Pain in Lower Abdomen, Sadness, Fatigue',
        'Increased appetite, Anxiety, Emotional lability',
        'Pain in Lower Abdomen, Headache, Decreased sexual drive',
        'Pain during sexual intercourse, Irritability/anger, Insomnia',
        'Nausea/Vertigo, Fatigue, Anxiety',
        'Sadness, Emotional lability, Irritability/anger',
        'Headache, Pain in Lower Abdomen, Decreased appetite',
        'Pain during sexual intercourse, Increased appetite, Fatigue',
        'Pain in Lower Abdomen, Digestive Problems, Sadness',
        'Insomnia, Fatigue, Decreased sexual drive',
        'Diarrhea, Urinary Pain, Decreased appetite'
    ],
    'Category': [
        'Abdominal Issues', 'Breast & Digestion', 'Digestion & Discomfort', 'Mental Health', 
        'Mental Health', 'Fatigue & Sexual Health', 'Mental Health', 'Abdominal & Mental Health', 
        'Breast Health', 'Sexual Health & Mental Health', 'Nausea & Headache', 'Digestion & Appetite', 
        'Mental Health', 'Pain & Sexual Health', 'Abdominal & Fatigue', 'Mental Health & Appetite', 
        'Abdominal & Sexual Health', 'Mental Health & Sexual Health', 'Fatigue & Mental Health', 
        'Mental Health', 'Abdominal & Appetite', 'Fatigue & Sexual Health', 'Abdominal & Digestion', 
        'Sexual Health & Fatigue', 'Diarrhea & Pain'
    ]
}

# Convert to DataFrame
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

# Updated symptom options based on the new symptoms
symptom_options = [
    'Pain in Lower Abdomen', 'Urinary Pain', 'Pain at Defecation', 'Nausea/Vertigo', 
    'Headache', 'Breast Tenderness', 'Digestive Problems', 'Diarrhea', 'Constipation', 
    'Discomfort due to vaginal bleeding', 'Feeling of being impure', 'Sadness', 
    'Emotional lability', 'Irritability/anger', 'Impulsiveness', 'Anxiety', 'Increased appetite', 
    'Decreased appetite', 'Insomnia', 'Hypersomnia', 'Fatigue', 'Decreased sexual drive', 
    'Concentration impairment', 'Pain during sexual intercourse'
]

# Collecting symptoms from the user using checkboxes
st.header("Please select your 3 most severe current symptoms:")
selected_symptoms = []

for symptom in symptom_options:
    if st.checkbox(symptom):
        selected_symptoms.append(symptom)

# When the user submits the symptoms
if st.button("Get Advice"):

    if selected_symptoms:
        # Non-AI-based Advice
        st.subheader("Advice based on selected symptoms (Non-AI):")
    
        # Rule-based advice (Non-AI)
        if 'Pain in Lower Abdomen' in selected_symptoms and 'Urinary Pain' in selected_symptoms:
            st.write("You may be experiencing urinary tract or abdominal issues. Consider seeing a doctor for further diagnosis.")
        elif 'Fatigue' in selected_symptoms and 'Pain during sexual intercourse' in selected_symptoms:
            st.write("These symptoms could indicate hormonal or sexual health issues. Please consult a healthcare provider.")
        elif 'Sadness' in selected_symptoms and 'Emotional lability' in selected_symptoms:
            st.write("You may be experiencing emotional distress. It could be helpful to talk to a counselor or a mental health professional.")
        elif 'Diarrhea' in selected_symptoms and 'Discomfort due to vaginal bleeding' in selected_symptoms:
            st.write("These symptoms may suggest a digestive or gynecological issue. A visit to a healthcare provider is recommended.")
        elif 'Nausea/Vertigo' in selected_symptoms and 'Headache' in selected_symptoms:
            st.write("You may be experiencing a migraine or dizziness. Hydration and rest could help, but it's advisable to consult a healthcare professional if symptoms persist.")
        elif 'Breast Tenderness' in selected_symptoms and 'Fatigue' in selected_symptoms:
            st.write("These symptoms may be associated with hormonal changes. If symptoms continue, consider seeing a healthcare provider.")
        elif 'Irritability/anger' in selected_symptoms and 'Anxiety' in selected_symptoms:
            st.write("You may be dealing with stress or anxiety. Consider practicing relaxation techniques or seeking help from a mental health professional.")
        elif 'Increased appetite' in selected_symptoms and 'Decreased sexual drive' in selected_symptoms:
            st.write("These symptoms may indicate hormonal imbalances or stress. A healthcare provider can help determine the underlying cause.")
        elif 'Decreased appetite' in selected_symptoms and 'Fatigue' in selected_symptoms:
            st.write("These symptoms could indicate stress, fatigue, or a possible underlying health issue. It’s best to speak to a healthcare provider.")
        elif 'Insomnia' in selected_symptoms and 'Fatigue' in selected_symptoms:
            st.write("You may be experiencing sleep disturbances that are affecting your energy levels. Consider improving sleep hygiene and consult a doctor if the issue persists.")
        elif 'Hypersomnia' in selected_symptoms and 'Fatigue' in selected_symptoms:
            st.write("Excessive sleepiness along with fatigue may suggest a sleep disorder or underlying health concern. Seek medical advice for a diagnosis.")
        elif 'Pain in Lower Abdomen' in selected_symptoms and 'Constipation' in selected_symptoms:
            st.write("These symptoms could suggest gastrointestinal issues. A balanced diet with fiber and hydration might help, but see a doctor if the symptoms persist.")
        elif 'Pain during sexual intercourse' in selected_symptoms and 'Pain in Lower Abdomen' in selected_symptoms:
            st.write("These symptoms may be related to gynecological issues, such as endometriosis or fibroids. Please see a gynecologist for further evaluation.")
        elif 'Concentration impairment' in selected_symptoms and 'Fatigue' in selected_symptoms:
            st.write("Difficulty concentrating along with fatigue can be signs of burnout or stress. Rest and self-care are essential, and you may need professional support.")
        elif 'Sadness' in selected_symptoms and 'Anxiety' in selected_symptoms:
            st.write("Sadness and anxiety can be indicative of mental health challenges. Seeking help from a counselor or mental health professional is advisable.")
        elif 'Feeling of being impure' in selected_symptoms and 'Irritability/anger' in selected_symptoms:
            st.write("Feelings of impurity and irritability can be associated with mental health or hormonal issues. It may be helpful to discuss these with a therapist or healthcare provider.")
        elif 'Breast Tenderness' in selected_symptoms and 'Headache' in selected_symptoms:
            st.write("These symptoms may indicate hormonal fluctuations, such as those before a menstrual period. If they persist, a healthcare provider can help.")
        elif 'Nausea/Vertigo' in selected_symptoms and 'Fatigue' in selected_symptoms:
            st.write("Nausea and fatigue could indicate a number of issues, including low blood sugar or vertigo. Please consult a healthcare professional for an accurate diagnosis.")
        elif 'Irritability/anger' in selected_symptoms and 'Pain in Lower Abdomen' in selected_symptoms:
            st.write("You may be experiencing symptoms related to hormonal changes or menstrual discomfort. Consider tracking your symptoms and discussing with your doctor.")
        elif 'Diarrhea' in selected_symptoms and 'Constipation' in selected_symptoms:
            st.write("Having both diarrhea and constipation can sometimes indicate a condition such as irritable bowel syndrome (IBS). Consult a gastroenterologist for a proper diagnosis.")
        else:
            st.write("Please consult a healthcare provider for a more accurate diagnosis.")

        if 'Pain in Lower Abdomen' in selected_symptoms:
            st.write("Pain in the lower abdomen can be caused by several factors such as menstrual cramps, gastrointestinal issues, or urinary tract infections. It is best to consult a healthcare provider for a proper diagnosis.")
        if 'Urinary Pain' in selected_symptoms:
            st.write("Urinary pain may indicate a urinary tract infection (UTI) or other bladder issues. Drinking plenty of water and seeing a doctor for further evaluation is recommended.")
        if 'Nausea/Vertigo' in selected_symptoms:
            st.write("Nausea and vertigo can be caused by a variety of conditions such as dehydration, inner ear problems, or vestibular disorders. Rest and hydration might help, but if symptoms persist, consult a healthcare provider.")
        if 'Headache' in selected_symptoms:
            st.write("Headaches can be triggered by various factors like stress, dehydration, or lack of sleep. Managing stress and staying hydrated can help alleviate mild headaches. If severe or recurring, consult a doctor.")
        if 'Breast Tenderness' in selected_symptoms:
            st.write("Breast tenderness can be linked to hormonal changes, menstruation, or pregnancy. If symptoms persist or are associated with other concerns, consider speaking with a healthcare provider.")
        if 'Digestive Problems' in selected_symptoms:
            st.write("Digestive problems such as bloating, constipation, or diarrhea can be related to diet or stress. A balanced diet and staying hydrated can often help. If the issue continues, consult a gastroenterologist.")
        if 'Diarrhea' in selected_symptoms:
            st.write("Diarrhea can result from infections, food intolerance, or digestive issues. Staying hydrated is important. If diarrhea lasts more than a few days, see a doctor.")
        if 'Constipation' in selected_symptoms:
            st.write("Constipation is often caused by dehydration, lack of fiber in the diet, or stress. Increase fiber intake and stay hydrated. If persistent, consider consulting a healthcare professional.")
        if 'Sadness' in selected_symptoms:
            st.write("Sadness can be a sign of emotional distress or depression. Talking to a counselor or a therapist can help you manage your feelings.")
        if 'Emotional lability' in selected_symptoms:
            st.write("Emotional lability refers to rapid changes in emotional state and could be linked to stress, anxiety, or hormonal changes. Consider mindfulness exercises or speaking with a therapist.")
        if 'Irritability/anger' in selected_symptoms:
            st.write("Irritability and anger can be triggered by stress, lack of sleep, or emotional exhaustion. Try practicing relaxation techniques like deep breathing or meditation.")
        if 'Anxiety' in selected_symptoms:
            st.write("Anxiety can manifest physically and emotionally. It may help to practice relaxation techniques, exercise, and seek professional help if the anxiety becomes overwhelming.")
        if 'Increased appetite' in selected_symptoms:
            st.write("Increased appetite can be a sign of stress, hormonal imbalances, or other health concerns. It's important to maintain a balanced diet and seek medical advice if the change is persistent.")
        if 'Decreased appetite' in selected_symptoms:
            st.write("Decreased appetite can be caused by stress, illness, or emotional distress. It is important to eat small, balanced meals, and consult a healthcare provider if the problem continues.")
        if 'Insomnia' in selected_symptoms:
            st.write("Insomnia can be due to stress, anxiety, or an underlying health condition. Improving your sleep hygiene and seeking professional help for persistent issues can be beneficial.")
        if 'Hypersomnia' in selected_symptoms:
            st.write("Hypersomnia, or excessive sleepiness, can be related to sleep disorders, depression, or other health concerns. Seek medical advice to rule out underlying conditions.")
        if 'Fatigue' in selected_symptoms:
            st.write("Fatigue can be caused by a variety of factors, including stress, sleep deprivation, or an underlying medical condition. Make sure you're getting adequate rest, and see a doctor if fatigue persists.")
        if 'Decreased sexual drive' in selected_symptoms:
            st.write("A decrease in sexual drive can be influenced by stress, hormonal imbalances, or emotional issues. It may help to discuss with a healthcare provider to determine the cause.")
        if 'Concentration impairment' in selected_symptoms:
            st.write("Difficulty concentrating can be linked to stress, lack of sleep, or mental fatigue. Ensuring you have a balanced routine and taking time for relaxation can help.")
        if 'Pain during sexual intercourse' in selected_symptoms:
            st.write("Pain during sexual intercourse can be related to hormonal issues, infections, or psychological factors. It's important to discuss this symptom with a gynecologist.")
        if 'Feeling of being impure' in selected_symptoms:
            st.write("Feelings of being impure or dirty can be associated with emotional distress, guilt, or stress. It might be helpful to speak with a therapist or counselor.")
        if 'Discomfort due to vaginal bleeding' in selected_symptoms:
            st.write("Vaginal bleeding can be linked to menstrual cycles, hormonal changes, or other gynecological issues. It's important to track your symptoms and consult a healthcare provider.")
        if 'Pain in Lower Abdomen' in selected_symptoms and 'Urinary Pain' in selected_symptoms:
            st.write("These combined symptoms may indicate a urinary tract infection (UTI) or gynecological issues. Please consult a healthcare provider for proper diagnosis and treatment.")
        if 'Fatigue' in selected_symptoms and 'Pain during sexual intercourse' in selected_symptoms:
            st.write("These symptoms might indicate hormonal imbalances, stress, or other health concerns. A healthcare provider will be able to guide you in determining the cause.")

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
        if category == 'Abdominal Issues':
            st.write("AI-based advice: It may be an issue with your abdominal region, possibly related to digestion or menstrual health. Consider a consultation with a doctor for further diagnosis.")
        elif category == 'Breast & Digestion':
            st.write("AI-based advice: You might be experiencing discomfort related to your digestive system or breast health. Consider monitoring your diet and seeking medical advice if symptoms persist.")
        elif category == 'Mental Health':
            st.write("AI-based advice: These symptoms might indicate a mental health issue, such as anxiety or stress. It's recommended to practice relaxation techniques and consult a mental health professional.")
        elif category == 'Sexual Health & Mental Health':
            st.write("AI-based advice: Your symptoms may be related to sexual health or emotional stress. It may be helpful to speak to a healthcare provider or therapist for further advice.")
        elif category == 'Fatigue & Sexual Health':
            st.write("AI-based advice: These symptoms could be linked to hormonal or sexual health issues. Consider seeing a doctor for a more thorough evaluation.")
        elif category == 'Digestion & Discomfort':
            st.write("AI-based advice: These symptoms may be due to digestive issues or menstrual problems. A balanced diet and medical consultation may help address these concerns.")
        else:
            st.write("AI-based advice: Please seek professional medical advice for more personalized recommendations.")
    else:
        st.write("Please select at least one symptom.")
