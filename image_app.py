# para rodar "streamlit run image_app.py"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import datetime
from transformers import pipeline
from datetime import datetime


plt.style.use('default')  
sns.set_palette("husl")  
sns.set_style("whitegrid")  

st.set_page_config(
    page_title="AI Image Classification Dashboard",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'analyzed_images' not in st.session_state:
    st.session_state.analyzed_images = []

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if 'classifier' not in st.session_state:
    st.session_state.classifier = None

@st.cache_resource 
def load_image_classifier():
    """Load and cache the image classification model"""
    try:
        classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        try:
            classifier = pipeline("image-classification", model="microsoft/resnet-50", device=-1)
            return classifier
        except:
            return None

def preprocess_image(image):
    """Preprocess image for classification"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def classify_image(image, classifier, top_k=5):
    """Classify a single image"""
    try:
        processed_image = preprocess_image(image)
        results = classifier(processed_image, top_k=top_k)
        return results
    except Exception as e:
        st.error(f"Error classifying image: {e}")
        return None

# TASK 12: Create prediction visualization chart
def create_prediction_chart(predictions):
    """Create a horizontal bar chart for predictions using matplotlib"""
    if not predictions:
        return None
    
    labels = [pred['label'] for pred in predictions]
    scores = [pred['score'] for pred in predictions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(labels, scores, color=plt.cm.viridis(np.array(scores)))
    
    # TODO: Customize the chart
    ax.set_xlabel('Confidence Score')  
    ax.set_title('Top Predictions')  
    ax.set_xlim(0, 1) 
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_analytics_dashboard(analyzed_images):
    """Create analytics visualizations using matplotlib"""
    if not analyzed_images:
        st.info("No analyzed images yet. Upload and classify some images first!")
        return
    
    all_predictions = []
    for img_data in analyzed_images:
        for pred in img_data['predictions']:
            all_predictions.append({
                'image_name': img_data['name'],
                'label': pred['label'],
                'score': pred['score'],
                'timestamp': img_data['timestamp']
            })
    
    df = pd.DataFrame(all_predictions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", len(analyzed_images))
    
    with col2:
        st.metric("Total Predictions", len(df))
    
    with col3:
        avg_confidence = df['score'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    with col4:
        top_class = df['label'].mode().iloc[0] if not df.empty else "None"
        st.metric("Most Common Class", top_class)
    
    col1, col2 =  st.columns(2)
    
    with col1:
        class_counts = df['label'].value_counts().head(10)
        
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        bars = ax1.barh(class_counts.index, class_counts.values, color='skyblue')
        ax1.set_xlabel('Count')
        ax1.set_title('Top 10 Predicted Classes')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center')
        
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.hist(df['score'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Score Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)

st.title("🖼️ AI Image Classification Dashboard")

st.markdown("""
Classify images using state-of-the-art AI models from Hugging Face. 
Upload your images to get instant predictions with confidence scores and detailed analytics.
""")

with st.sidebar:
    st.header("🛠️ Configuration")
    
    model_option = st.selectbox(
        "Choose Model",
        [
            "google/vit-base-patch16-224 (Recommended)",
            "microsoft/resnet-50 (Faster)"
        ]
    )
    
    st.subheader("Classification Options")
    top_k = st.slider("Top K Predictions", 1, 10, 5)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.1)
    
    if st.button("🚀 Load Model", type="primary"):
        with st.spinner("Loading AI model..."):
            st.session_state.classifier =  load_image_classifier()
            st.session_state.model_loaded =  True
        
        if st.session_state.classifier:
            # TODO: Show success message
            st.success("✅ Model loaded successfully!")
        else:
            # TODO: Show error message
            st.error("❌ Failed to load model")


tab1, tab2, tab3 = st.tabs(["📷 Single Image", "📂 Image History", "📊 Results & Analytics"])

with tab1:
    st.header("Single Image Classification")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔍 Classify Image", type="primary", disabled=not st.session_state.model_loaded):
                if not st.session_state.classifier:
                    st.error("Please load the model first!")
                else:
                    with st.spinner("Classifying image..."):
                        predictions = classify_image(image, st.session_state.classifier, top_k)
                        
                        if predictions:
                            image_data = {
                                'name': uploaded_file.name,
                                'predictions': predictions,
                                'timestamp': datetime.now(),
                                'image': image
                            }
                            st.session_state.analyzed_images.append(image_data)
                            
                            st.subheader("🎯 Predictions")
                            
                            for i, pred in enumerate(predictions):
                                confidence_color = "🟢" if pred['score'] > confidence_threshold else "🟡"
                                st.write(f"{i+1}. {confidence_color} **{pred['label']}** - {pred['score']:.2%}")
                            
                            chart = create_prediction_chart(predictions)
                            if chart:
                                st.pyplot(chart)

with tab2:
    st.header("Classification History")
    
    if st.session_state.analyzed_images:
        st.write(f"Total classified images: {len(st.session_state.analyzed_images)}")
        
        for idx, img_data in enumerate(reversed(st.session_state.analyzed_images)):  
            with st.expander(f"{img_data['name']} - {img_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(img_data['image'], caption=img_data['name'], use_container_width=True)
                
                with col2:
                    st.write("**Top Predictions:**")
                    for i, pred in enumerate(img_data['predictions'][:3]):  
                        st.write(f"{i+1}. **{pred['label']}** - {pred['score']:.2%}")
    else:
        st.info("No images classified yet. Go to the Single Image tab to start classifying!")

with tab3:
    st.header("Results & Analytics")
    create_analytics_dashboard(st.session_state.analyzed_images)
    
    if st.session_state.analyzed_images:
        st.markdown("### 📋 Detailed Results")
        
        detailed_results = []
        for img_data in st.session_state.analyzed_images:
            top_pred = img_data['predictions'][0]
            detailed_results.append({
                'Image': img_data['name'],
                'Top Prediction': top_pred['label'],
                'Confidence': f"{top_pred['score']:.2%}",
                'Timestamp': img_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        results_df = pd.DataFrame(detailed_results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # TASK 20: Download functionality
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"image_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # TASK 21: Clear results functionality
        if st.button("🗑️ Clear All Results", type="secondary"):
            st.session_state.analyzed_images = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit, Matplotlib, and Hugging Face Transformers<br>
    Perfect for learning AI deployment and image classification!
</div>
""", unsafe_allow_html=True)