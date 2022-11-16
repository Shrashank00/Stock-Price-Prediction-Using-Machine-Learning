import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


st.title('Stock Price Prediction')


user_input=st.text_input('Enter Stock Ticker','GOOG')

start_date=st.date_input('Pick a start date')
end_date=st.date_input('Pick a end date')


df=data.DataReader(user_input,'yahoo',start_date,end_date)

start_year=start_date.year
end_year=end_date.year
#Describing the data
st.subheader('Data from '+str(start_year)+'-'+str(end_year))
st.write(df.describe())

st.markdown("""<hr>""",True)
##############################################
st.subheader('Closing Price vs Time chart')
st.line_chart(df.Close)

st.markdown("""<hr>""",True)

st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,label="Closing Price ")
plt.plot(ma100,label="MA100")
plt.legend()
st.pyplot(fig)

st.markdown("""<hr>""",True)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,label="Closing Price ")
plt.plot(ma100,label="MA100")
plt.plot(ma200,label="MA200")
plt.legend()
st.pyplot(fig)


data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)



#Load my model

model=load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)



x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)


scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

st.markdown("""<hr>""",True)

st.header('Predictions vs Original Price')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label="Predicted Price")
plt.xlabel('Time (days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
##############################################
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
past100=int(len(data_testing))-100
data_testing=scaler.fit_transform(data_testing)

x_input=np.array(data_testing[past100:]).reshape(1,-1)
inputx=scaler.fit_transform(x_input)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

#from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(int(len(temp_input))>100):

        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))

        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
       
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)

        temp_input.extend(yhat[0].tolist())
       
        lst_output.extend(yhat.tolist())
        i=i+1

lst_output=np.array(lst_output)

lst_output=scale_factor*lst_output

new_predict=np.concatenate((y_predicted,lst_output))

st.markdown("""<hr>""",True)

st.subheader("Future 30 days trend")

fig3=plt.figure(figsize=(12,6))
plt.plot(new_predict[len(new_predict)-30:],'g',label="Future Price")
plt.xlabel('Time (days)')
plt.ylabel('Price (in $)')
plt.legend()
st.pyplot(fig3)

st.markdown("""<hr>""",True)

st.subheader("Predicted price")
fig4=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
#plt.plot(y_predicted,'r',label="Predicted Price")
plt.plot(new_predict,'r',label="Future Price")
plt.xlabel('Time (days)')
plt.ylabel('Price (in $)')
plt.legend()
st.pyplot(fig4)
