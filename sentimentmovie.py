import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
#Load reviews from my directory
def load_reviews(directory,sentiment):
    reviews=[]
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory,filename),'r',encoding='utf-8') as file:
                reviews.append((file.read(),sentiment))
    return reviews
#path of training data
t_pos=r'C:\Users\risha\Downloads\down\aclImdb_v1\aclImdb\train\pos'
t_neg=r'C:\Users\risha\Downloads\down\aclImdb_v1\aclImdb\train\neg'
#Load Data
pos_review=load_reviews(t_pos,1)
neg_review=load_reviews(t_neg,0)
#Create a DataFrame
reviews_df=pd.DataFrame(pos_review+neg_review,columns=['review','sentiment'])

X_train, X_test, Y_train, Y_test=train_test_split(reviews_df['review'],reviews_df['sentiment'],test_size=0.2,random_state=42)

#vectorize the text data
vectorizer=TfidfVectorizer(stop_words='english',max_features=5000)
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)

#train the model
model=LogisticRegression()
model.fit(X_train_vec,Y_train)

#user input sentiment prediction

def p_sentiment(review,model,vectorizer):
    review_vec=vectorizer.transform([review])
    prediction=model.predict(review_vec)[0]
    return "Positive" if prediction == 1 else "Negative"

#user input
print("sentiment analysis: enter a movie review(or type 'exit'):")
while True:
    user_in=input("Your review: ")
    if user_in.lower() == 'exit':
        print(" Exiting...")
        break
    sentiment=p_sentiment(user_in,model,vectorizer)
    print(f"Predicted Sentiment: {sentiment} ")
""" 
   # Predict on the test set
    Y_pred = model.predict(X_test_vec)

    # Accuracy score
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Classification report
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred))
"""
