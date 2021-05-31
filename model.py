import numpy as np
import pandas as pd
import pickle

def predict_value(user_input):

    pickled_tfidf_vectorizer = pickle.load(open('models/tranform.pkl','rb'))
    pickled_model = pickle.load(open('models/nlp_model.pkl','rb'))
    pickled_user_final_rating = pickle.load(open('models/user_final_rating.pkl','rb'))        
    pickled_mapping = pickle.load(open('models/prod_id_name_mapping.pkl','rb')) 
    pickled_reviews_data = pickle.load(open('models/MainData.pkl','rb'))

    #print(pickled_user_final_rating)
    
    #recommendations = pd.DataFrame(pickled_user_final_rating.loc[user_input]).reset_index()
    #recommendations.columns=['id','user_pred_rating']
    #recommendations= recommendations.sort_values(by='user_pred_rating',ascending=False)[0:20]
    

    recommendations = pd.DataFrame(pickled_user_final_rating.loc[user_input]).reset_index()
    recommendations.rename(columns={recommendations.columns[1]: "pred_rating" }, inplace = True)
    recommendations = recommendations.sort_values(by='pred_rating', ascending=False)[0:20]
    
    pickled_reviews_data.rename(columns={pickled_reviews_data.columns[0]: "id" }, inplace = True)

    recommendations = pd.merge(recommendations,pickled_mapping, left_on="id", right_on="id", how = "left")
    
    improved_recommendations= pd.merge(recommendations,pickled_reviews_data[['id','reviews_text']], left_on='id', right_on='id', how = 'left')
    test_data_for_user = pickled_tfidf_vectorizer.transform(improved_recommendations['reviews_text'].values.astype('U'))
    
    sentiment_prediction_for_user = pickled_model.predict(test_data_for_user)
    sentiment_prediction_for_user = pd.DataFrame(sentiment_prediction_for_user, columns=['Predicted_Sentiment'])

    improved_recommendations= pd.concat([improved_recommendations, sentiment_prediction_for_user], axis=1)
    
    a=improved_recommendations.groupby('id')
    b=pd.DataFrame(a['Predicted_Sentiment'].count()).reset_index()
    b.columns = ['id', 'Total_reviews']        
    c=pd.DataFrame(a['Predicted_Sentiment'].sum()).reset_index()
    c.columns = ['id', 'Total_predicted_positive_reviews']
    
    improved_recommendations_final=pd.merge( b, c, left_on='id', right_on='id', how='left')
    
    improved_recommendations_final['Positive_sentiment_rate'] = improved_recommendations_final['Total_predicted_positive_reviews'].div(improved_recommendations_final['Total_reviews']).replace(np.inf, 0)
    
    improved_recommendations_final= improved_recommendations_final.sort_values(by=['Positive_sentiment_rate'], ascending=False )
    improved_recommendations_final=pd.merge(improved_recommendations_final, pickled_mapping, left_on='id', right_on='id', how='left')
    
    name_display= improved_recommendations_final.head(5)
    name_display= name_display['name']
    
    output = name_display.to_list()
    output.insert(0,": ")
    output="--> ".join(output)
    #print(output)
    return 'Top 5 recommendations: {}'.format(output)

    
    
    
    
    
    
    
