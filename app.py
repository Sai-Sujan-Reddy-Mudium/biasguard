import streamlit as st
import pandas as pd
from google import genai

# Set the page configuration (tab title and layout)
st.set_page_config(page_title="BiasGuard", layout="wide")

# Add a main title and description
st.title("⚖️ BiasGuard: AI Fairness Auditor")
st.write("Upload a dataset to evaluate algorithmic bias and calculate the Disparate Impact Ratio.")

st.divider() # Draws a horizontal line

# 1. Create a file uploader widget
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

# 2. Check if a file was uploaded
if uploaded_file is not None:
    # Read the file using pandas
    df = pd.read_csv(uploaded_file)

    # Strip whitespace (from Phase 2)
    df_obj = df.select_dtypes(include=['object', 'str'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    st.success("File uploaded and cleaned successfully!")

    # Display the first 10 rows in an interactive table
    st.write("### Data Preview")
    st.dataframe(df.head(10))
    st.divider()
    st.write("### Configure Audit Parameters")

    # Create two columns for a cleaner UI
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**1. Select the Target Outcome** (e.g., Income, Loan Approval)")
        target_col = st.selectbox("Target Column", df.columns)
        # Find unique values in the selected column for the next dropdown
        favorable_val = st.selectbox("Which value represents the FAVORABLE outcome?", df[target_col].unique())

    with col2:
        st.markdown("**2. Select the Protected Class** (e.g., Sex, Race)")
        protected_col = st.selectbox("Protected Column", df.columns)
        privileged_val = st.selectbox("Which group is historically PRIVILEGED?", df[protected_col].unique())
        unprivileged_val = st.selectbox("Which group is historically UNPRIVILEGED?", df[protected_col].unique())

    st.divider()

    # The Action Button
    if st.button("Run Fairness Audit", type="primary"):

        # --- THE MATH ENGINE (Adapted from Phase 2) ---
        privileged_group = df[df[protected_col] == privileged_val]
        unprivileged_group = df[df[protected_col] == unprivileged_val]

        priv_favorable = len(privileged_group[privileged_group[target_col] == favorable_val])
        unpriv_favorable = len(unprivileged_group[unprivileged_group[target_col] == favorable_val])

        # Error handling for empty groups
        if len(privileged_group) == 0 or len(unprivileged_group) == 0:
            st.error("Error: One of the selected groups has no data. Check your selections.")
        else:
            priv_rate = priv_favorable / len(privileged_group)
            unpriv_rate = unpriv_favorable / len(unprivileged_group)

            if priv_rate == 0:
                 st.error("Error: The privileged group has 0 favorable outcomes. Cannot divide by zero.")
            else:
                dir_score = unpriv_rate / priv_rate

                # --- UI DISPLAY ---
                st.write("### Audit Results")

                # Create 3 columns for metrics
                m1, m2, m3 = st.columns(3)
                m1.metric(label=f"{privileged_val} Success Rate", value=f"{priv_rate*100:.1f}%")
                m2.metric(label=f"{unprivileged_val} Success Rate", value=f"{unpriv_rate*100:.1f}%")
                m3.metric(label="Disparate Impact Ratio", value=f"{dir_score:.3f}")

                # Legal interpretation
                if dir_score < 0.80:
                    st.error(f"🚨 **BIAS DETECTED:** The DIR is {dir_score:.3f}, which is below the 0.80 legal threshold. This indicates statistical bias against {unprivileged_val}.")
                else:
                    st.success(f"✅ **PASS:** The DIR is {dir_score:.3f}, which meets the 0.80 threshold. No statistical bias detected in this specific split.")
                
                # --- AI REPORT GENERATION ---
                st.divider()
                st.write("### 🤖 AI Auditor Report")

                # Create a loading spinner while waiting for Google's servers
                with st.spinner("Gemini is analyzing the statistical bias..."):
                        
                        prompt = f"""
                        Act as an expert AI Ethics and Compliance Auditor. 
                        I have analyzed a dataset predicting {target_col}. 
                        The protected class is {protected_col}. 
                        The privileged group is {privileged_val} with a success rate of {priv_rate*100:.1f}%. 
                        The unprivileged group is {unprivileged_val} with a success rate of {unpriv_rate*100:.1f}%. 
                        The Disparate Impact Ratio (DIR) is {dir_score:.3f}.
                        
                        Write a 3-paragraph executive summary:
                        1. Explain what this specific DIR score means in plain English.
                        2. State whether this passes or fails the standard 0.80 legal threshold.
                        3. Suggest two concrete technical steps engineers can take to mitigate this bias in the dataset.
                        """
                        
                        try:
                            # 1. Initialize the modern client with your secret key
                            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                            
                            # 2. Call the current 2.5-flash model
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=prompt
                            )
                            
                            # 3. Display the response on the web page
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Failed to connect to Google API: {e}")