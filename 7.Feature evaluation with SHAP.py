import shap
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()


CD_model = load_model('CD_model.h5')
CD_model.summary()

background = X_train
background.shape


explainer = shap.DeepExplainer(CD_model,background)
shap_values = explainer.shap_values(X_train)  


shap_values_save = pd.DataFrame(shap_values[0])
feature_importance = shap_values_save.abs().mean().sort_values(ascending = False)

# summarize the effects of all the features
shap.summary_plot(shap_values[0],
                  X_train,
                  feature_names=marker.index,
                  plot_type = 'bar',
#                   plot_type = 'dot',
                  max_display=20)