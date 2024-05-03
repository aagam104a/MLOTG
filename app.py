import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from abc import ABC, abstractmethod

class EncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df, selected_features):
        pass

class LabelEncodingStrategy(EncodingStrategy):
    def encode(self, df, selected_features):
        label_encoder = LabelEncoder()
        df_encoded = df.copy()
        for feature in selected_features:
            df_encoded[feature] = label_encoder.fit_transform(df_encoded[feature])
        return df_encoded

class OneHotEncodingStrategy(EncodingStrategy):
    def encode(self, df, selected_features):
        df_encoded = df.copy()
        for feature in selected_features:
            one_hot_encoder = OneHotEncoder(sparse_output=False)  # Updated here
            one_hot_encoded = one_hot_encoder.fit_transform(df_encoded[[feature]])
            one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=[f"{feature}_{i}" for i in range(one_hot_encoded.shape[1])])
            df_encoded = pd.concat([df_encoded, one_hot_encoded_df], axis=1)
            df_encoded.drop(columns=[feature], inplace=True)
        return df_encoded

class Encoder:
    def __init__(self, strategy):
        self._strategy = strategy

    def encode(self, df, selected_features):
        return self._strategy.encode(df, selected_features)

def main():
    st.title('ML On The Go')

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        st.write("Original Dataset:")
        df = pd.read_csv(uploaded_file)
        st.write(df)

        selected_features = st.multiselect("Select features for encoding", df.columns)

        if selected_features:
            encoding_method = st.selectbox("Select Encoding Method", ["Label Encoding", "One-Hot Encoding"])

            if encoding_method == "Label Encoding":
                encoder = Encoder(LabelEncodingStrategy())
                encoded_df = encoder.encode(df, selected_features)
                st.write("Label Encoded Dataset:")
                st.write(encoded_df)
            elif encoding_method == "One-Hot Encoding":
                encoder = Encoder(OneHotEncodingStrategy())
                encoded_df = encoder.encode(df, selected_features)
                st.write("One-Hot Encoded Dataset:")
                st.write(encoded_df)

if __name__ == "__main__":
    main()
