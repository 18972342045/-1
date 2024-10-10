import streamlit as st 
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# 设置中文字体和绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载预训练的模型和标准化器
model_path = "C:/Users/a1897/random_forest_model.pkl"  # 替换为你的模型路径
scaler_path = "C:/Users/a1897/scaler.pkl"  # 替换为你的标准化器路径

# 加载模型
with open(model_path, "rb") as model_file:
    classifier = pickle.load(model_file)

# 加载标准化器
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# 定义预测函数
def predict(input_data):
    # 标准化输入数据
    input_data_scaled = scaler.transform(input_data)

    # 进行预测
    prediction = classifier.predict(input_data_scaled)
    prediction_proba = classifier.predict_proba(input_data_scaled)
    
    return prediction[0], prediction_proba

# Streamlit主界面
def main():
    st.title("出院后 1 年内 MACE 再次发生预测模型")

    # 创建输入框让用户输入数据
    age = st.number_input("年龄", min_value=0, max_value=120, value=50, step=1)
    height = st.number_input("身高 (cm)", min_value=100, max_value=250, value=170, step=1)
    weight = st.number_input("体重 (kg)", min_value=30, max_value=200, value=70, step=1)
    systolic_bp = st.number_input("收缩压 (mmHg)", min_value=50, max_value=200, value=120, step=1)
    diastolic_bp = st.number_input("舒张压 (mmHg)", min_value=50, max_value=120, value=80, step=1)
    
    hypertension_history = st.selectbox("高血压病史", ("是", "否"))
    diabetes_history = st.selectbox("糖尿病史", ("是", "否"))

    # 将输入数据组合成DataFrame
    input_data = pd.DataFrame([[age, height, weight, systolic_bp, diastolic_bp, 
                                1 if hypertension_history == '是' else 0, 
                                1 if diabetes_history == '是' else 0]],
                              columns=["年龄", "身高", "体重", "收缩压", "舒张压", "高血压病史", "糖尿病史"])
    
    # 预测按钮
    if st.button("预测"):
        prediction, probability = predict(input_data.values)

        # 根据预测结果进行解释
        if prediction == 1:
            st.write(f"预测的概率: 患者在出院后一年内再次发生MACE的概率为 {probability[0][1]:.2%}")
        else:
            st.write(f"预测的概率: 患者在出院后一年内未再次发生MACE的概率为 {probability[0][0]:.2%}")

        # 使用 SHAP 解释模型输出
        explainer = shap.TreeExplainer(classifier)
        input_data_scaled = scaler.transform(input_data)  # 标准化输入数据
        shap_values = explainer.shap_values(input_data_scaled)

        st.subheader("SHAP值可视化")
        shap.initjs()
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], input_data_scaled))

# 将SHAP plot对象转换为HTML格式
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

if __name__ == '__main__':
    main()
