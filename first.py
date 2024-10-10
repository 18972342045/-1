#本代码只是一个测试示例代码，根据不同预测模型任务，加载不同的模型，同时需要将代码中涉及sepal_length, sepal_width, petal_length, petal_width的地方修改为自己需要的变量名
#切记变量名需要和训练模型时的变量名一一对应

import numpy as np
import pickle
import streamlit as st
import shap
import matplotlib
import streamlit.components.v1 as components

matplotlib.use('TkAgg')

#将文件中的路径修改为本地模型文件，通常为pkl文件,Windows电脑记得写绝对路径
pickle_in1 = open("C:/Users/a1897/Downloads/classifier1.pkl", "rb")
classifier1 = pickle.load(pickle_in1)  #加载分类器

#定义prediction1函数用来预测
def prediction1(sepal_length, sepal_width, petal_length, petal_width):   #需要将sepal_length, sepal_width, petal_length, petal_width修改为你自己需要的变量名
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    y_pred = classifier1.predict(X)
    return y_pred[0]

#使用st_shap函数将shap.Plot对象转换为HTML
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def main():
    st.title("鸢尾花品种预测")
    # 获取用户输入
    # 根据实际变量建立网页上对应的变量交互输入
    sepal_length1 = st.number_input("花萼长度", min_value=0.0, max_value=10.0, value=5.8, step=0.1)
    sepal_width1 = st.number_input("花萼宽度", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length1 = st.number_input("花瓣长度", min_value=0.0, max_value=10.0, value=3.8, step=0.1)
    petal_width1 = st.number_input("花瓣宽度", min_value=0.0, max_value=10.0, value=1.1, step=0.1)

    if st.button("预测"):
        prediction = prediction1(float(sepal_length1), float(sepal_width1), float(petal_length1), float(petal_width1))
        
        # 显示预测结果
        st.write(f"预测的鸢尾花品种: {prediction}")
        shap.initjs()
        data_for_prediction = np.array([[sepal_length1, sepal_width1, petal_length1, petal_width1]])
        prediction_proba = classifier1.predict_proba (data_for_prediction)
        st.subheader ( 'Prediction' )
        aki_probability = prediction_proba[0][1] * 100
        st.write ( f"Based on feature values, predicted possibility of {prediction} is {aki_probability:.2f}%" )
        # 单个样本的SHAP解释
        explainer = shap.TreeExplainer(classifier1)
        shap_values = explainer.shap_values(data_for_prediction)
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction))
if __name__ == '__main__':
    main()
    


