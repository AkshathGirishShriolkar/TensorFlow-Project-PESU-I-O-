# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import keras.backend as K

# %%
df=pd.read_csv(r'bitcoin.csv',
               parse_dates=["Date"],
               index_col=['Date'])
df.head()

# %%
bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
bitcoin_prices.head()

# %%
timesteps=bitcoin_prices.index.to_numpy()
prices=bitcoin_prices.Price.to_numpy()
bitcoin_prices.plot(figsize=(10,7))
plt.ylabel('BTC Price')
plt.title('price of bitcoin from 1st oct 2013 to 18th may 2021',fontsize=16)
plt.legend(fontsize=14)

# %%
timesteps=bitcoin_prices.index.to_numpy()
prices=bitcoin_prices.Price.to_numpy()


# %%
split_size=int(0.8 * len(prices))
X_train, y_train=timesteps[:split_size],prices[:split_size]
X_test, y_test = timesteps[split_size:],prices[split_size:]


# %%
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

# %%
def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
    plt.plot(timesteps[start:], values[start:], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend(fontsize=14) # make label bigger
        plt.grid(True)

plt.figure(figsize=(10,7))
plot_time_series(X_train,y_train,label='Train data')
plot_time_series(X_test, y_test, label='Test data')

# %%
def mean_absolute_scaled_error(y_true, y_pred):
    '''
    No seasonality is assumed
    '''
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
    return mae / mae_naive_no_season

# %%
# Create loss function to return all loss metrics in dictionary format
def evaluate_preds(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mase = mean_absolute_scaled_error(y_true, y_pred)

    return {"mase",mase.numpy()}

# %%
HORIZON=1
WINDOW_SIZE=7
def get_labelled_windows(x,horizon=HORIZON):
  return x[:,:-horizon],x[:,-horizon:]

# %%
def make_windows(x,window_size=7,horizon=1):
  """
  """
  window_step=np.expand_dims(np.arange(window_size+horizon),axis=0)
  window_indexes=window_step+np.expand_dims(np.arange(len(x)-(window_size+horizon-1)),axis=0).T
  windowed_array=x[window_indexes]
  windows,labels=get_labelled_windows(windowed_array)
  return windows,labels
full_windows, full_labels=make_windows(prices)
len(full_windows), len(full_labels)

# %%
for i in range(3):
  print(f'window: {full_windows[i]} -> Label: {full_labels[i]}')

# %%
def make_train_test_splits(windows, labels, test_split=0.2):
  split_size = int(len(windows)* (1-test_split))
  train_windows=windows[:split_size]
  train_labels=labels[:split_size]
  test_windows=windows[split_size:]
  test_labels=labels[split_size:]
  return train_windows, test_windows, train_labels,test_labels

train_windows, test_windows, train_labels,test_labels= make_train_test_splits(full_windows, full_labels)
len(train_windows),  len(full_windows), len(test_windows)

# %%
horizonss=1

# %%
train_windows.shape[1]

# %%
inputs = tf.keras.layers.Input(shape=(7,), dtype=tf.float64)
x = tf.keras.layers.Dense(124,activation='relu')(inputs)
x = tf.keras.layers.Dense(124,activation='relu')(x)
x = tf.keras.layers.Dense(124,activation='relu')(x)
outputs = tf.keras.layers.Dense(7+horizonss, activation='linear')(x)


block = tf.keras.Model(inputs, outputs)
block1 = tf.keras.Model(inputs, outputs)
block2 = tf.keras.Model(inputs, outputs)
block3 = tf.keras.Model(inputs, outputs)
block4 = tf.keras.Model(inputs, outputs)
block5 = tf.keras.Model(inputs, outputs)
block6 = tf.keras.Model(inputs, outputs)
block7 = tf.keras.Model(inputs, outputs)
block8 = tf.keras.Model(inputs, outputs)
block9 = tf.keras.Model(inputs, outputs)
block10 = tf.keras.Model(inputs, outputs)
block11 = tf.keras.Model(inputs, outputs)
block12 = tf.keras.Model(inputs, outputs)
block13 = tf.keras.Model(inputs, outputs)
block14 = tf.keras.Model(inputs, outputs)
block15 = tf.keras.Model(inputs, outputs)
block16 = tf.keras.Model(inputs, outputs)
block17 = tf.keras.Model(inputs, outputs)
block18 = tf.keras.Model(inputs, outputs)
block19 = tf.keras.Model(inputs, outputs)
block20 = tf.keras.Model(inputs, outputs)
block21 = tf.keras.Model(inputs, outputs)
block22 = tf.keras.Model(inputs, outputs)
block23 = tf.keras.Model(inputs, outputs)
block24 = tf.keras.Model(inputs, outputs)
block25 = tf.keras.Model(inputs, outputs)
block26 = tf.keras.Model(inputs, outputs)
block27 = tf.keras.Model(inputs, outputs)
block28 = tf.keras.Model(inputs, outputs)
block29 = tf.keras.Model(inputs, outputs)
block.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block1.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block2.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block3.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block4.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block5.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block6.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block7.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block8.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block9.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block10.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block11.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block12.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block13.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block13.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block14.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block15.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block16.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block17.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block18.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block19.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block20.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block21.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block22.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block23.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block24.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block25.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block26.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block27.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block28.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block29.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
block.summary()
'''
block.fit(train_dataset,
           epochs=1,
           validation_data=val_dataset)
'''
b=[block1, block2, block3, block4,block5, block6, block7, block8, block9, block10, block11, block12, block13, block14, block15, block16, block17, block18, block19, block20, block21, block22, block23, block24, block25, block26, block27, block28, block29 ]

# %%
inputss=tf.keras.layers.Input(shape=(7,))
l=block(inputss)

print(inputss.shape)
qwerty=tf.keras.layers.Dense(inputss.shape[1],activation='relu')(inputss)
qwe=tf.keras.layers.Dense(inputss.shape[1],activation='relu')(qwerty)
l=block(qwe)
print(type(l))

# %%
def TSeval(): 
    inputss=tf.keras.layers.Input(shape=(7,))
    l1=block(inputss)
    split_layer = tf.split(l1, num_or_size_splits=8)
    print(split_layer)
    #qwerty=tf.keras.layers.Dense(l1.shape[1],activation='relu')(l1)

    l=l1.numpy().tolist()
    FC=tf.Tensor((l.numpy())[2224:len(l)])# the element whose index is 2224. but 
    BC=tf.Tensor((l.numpy())[0:2224,:-horizonss])    # 0 to 2223 index. but elements is from 1 to 2224
    print(BC.shape)
    print(FC.shape)
    
    inp= tf.keras.layers.subtract([inputss, BC])
    ans=FC
    for i in b:
        l=i(inp)
        FC=tf.Tensor((l.tolist())[2224:len(l)])
        BC=tf.Tensor((l.tolist())[0:2224,:-horizonss])
        inp= tf.keras.layers.subtract([inp, BC])
        ans= tf.keras.layers.add([ans, FC])
    finalmodel=tf.keras.models.Model(inputss,ans)
    finalmodel.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
    finalmodel.fit(train_windows,train_labels, epochs=500, validation_data=(test_windows,test_labels), batch_size=128)
    return (finalmodel)


# %%
m=TSeval()
m.predict(test_windows)

# %%
x=tf.keras.layers.Dense(24)()

# %%
TSeval(train_windows)           

# %%



