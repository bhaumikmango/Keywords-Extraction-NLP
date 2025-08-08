import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Keyword Extractor using NLP",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .keyword-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .score-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        float: right;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
        padding: 1rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# Download NLTK data if not already present
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_data()

# Load models
@st.cache_resource
def load_models():
    try:
        with open('Count_Vector.pkl', 'rb') as f:
            cv = pickle.load(f)
        with open('TFIDF_Transformer.pkl', 'rb') as f:
            tfidf_trans = pickle.load(f)
        with open('Feature_Names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return cv, tfidf_trans, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please make sure you have run train.py and have the .pkl files in your directory.")
        return None, None, None

# Text preprocessing function
def preprocessing_text(txt):
    stop_words = set(stopwords.words('english')) 
    new_words = ['fig', 'figure', 'image', 'sample', 'using',
                 'show', 'result', 'large', 'also', 'one',
                 'two', 'three', 'four', 'five', 'six', 'seven',
                 'eight', 'nine']
    stop_words = list(stop_words.union(new_words))
    
    txt = txt.lower()
    txt = re.sub(r'<.*?>', ' ', txt)
    txt = re.sub(r'[^a-zA-Z]', ' ', txt)
    txt = nltk.word_tokenize(txt)
    txt = [word for word in txt if word not in stop_words]
    txt = [word for word in txt if len(word)>=3]
    stemming = PorterStemmer()
    txt = [stemming.stem(word) for word in txt]
    lmtr = WordNetLemmatizer()
    txt = [lmtr.lemmatize(word) for word in txt]
    return ' '.join(txt)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

def get_keywords(text, cv, tfidf_trans, feature_names, topn=10):
    processed_text = preprocessing_text(text)
    tf_idf_vector = tfidf_trans.transform(cv.transform([processed_text]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_names, sorted_items, topn)
    return keywords

def create_wordcloud(keywords):
    if not keywords:
        return None
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=50
    ).generate_from_frequencies(keywords)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

def create_bar_chart(keywords):
    if not keywords:
        return None
    
    words = list(keywords.keys())
    scores = list(keywords.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=scores[::-1],  # Reverse for horizontal bar
            y=words[::-1],   # Reverse for horizontal bar
            orientation='h',
            marker=dict(
                color=scores[::-1],
                colorscale='viridis',
                showscale=True
            ),
            text=[f'{score:.3f}' for score in scores[::-1]],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Keyword Importance Scores',
        xaxis_title='TF-IDF Score',
        yaxis_title='Keywords',
        height=400,
        margin=dict(l=100, r=50, t=50, b=50)
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">Keyword Extractor using NLP</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extract meaningful keywords from any text using advanced TF-IDF analysis</p>', unsafe_allow_html=True)
    
    # Load models
    cv, tfidf_trans, feature_names = load_models()
    
    if cv is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    num_keywords = st.sidebar.slider("Number of keywords to extract", 5, 20, 10)
    
    st.sidebar.header("📊 Visualization Options")
    show_wordcloud = st.sidebar.checkbox("Show Word Cloud", True)
    show_bar_chart = st.sidebar.checkbox("Show Bar Chart", True)
    
    st.sidebar.header("🔮 Use cases")
    st.sidebar.info("1. Academic Research & Publishing")
    st.sidebar.info("2. Content Management & SEO")
    st.sidebar.info("3. Market Research & Competitive Intelligence")
    st.sidebar.info("4. Legal & Compliance")
    st.sidebar.info("5. News & Media Monitoring")
    st.sidebar.info("6. E-commerce & Product Catalogs")
    st.sidebar.info("7. Customer Support & Feedback Analysis")
    st.sidebar.info("8. Healthcare & Medical Research")
    st.sidebar.info("9. Recruitment & HR Tech")
    st.sidebar.info("10. Real Estate Platforms")

    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        "This app uses TF-IDF (Term Frequency-Inverse Document Frequency) "
        "to extract the most important keywords from your text. "
        "The model was trained on academic papers and works best with "
        "formal, technical, or academic content."
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter your text")
        
        # Sample texts
        sample_texts = {
            "Select a sample...": "",
            "Machine Learning Research": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
            "Climate Change Article": "We often think about human-induced climate change as something that will happen in the future, but it is happening now. Ecosystems and people in the United States and around the world are affected by the ongoing process of climate change today. A collage of typical climate and weather-related events: floods, heatwaves, drought, hurricanes, wildfires and loss of glacial ice. A collage of events related to climate and weather: loss of glacial ice, wildfires, hurricanes, floods, and drought. (Image credit: NOAA) Climate change affects the environment in many different ways, including rising temperatures, sea level rise, drought, flooding, and more. These events affect things that we depend upon and value, like water, energy, transportation, wildlife, agriculture, ecosystems, and human health. Our changing climate NOAA monitors weather and climate around the world. Here are some of the ways that climate change is affecting our planet. You can explore more at NOAA’s Global Climate Dashboard. Global temperatures increased. Temperature increased about 2°F (1.1°C) from 1850 to 2023. Updated January 2024. Sea level rise has sped up. Global average sea level has risen 8–9 inches (21–24 centimeters) since 1880. Updated April 2022. Glaciers are shrinking  Climate reference glaciers tracked by the World Glacier Monitoring Service have lost ice for the past 36 years in a row. Updated May 2024. Arctic sea ice is decreasing. Between 1979 and 2021, the Arctic Ocean lost sea ice at an average of 31,100 square miles, an area the size of South Carolina, per year. Updated October 2022. Atmospheric carbon dioxide is increasing. There is more carbon dioxide in the atmosphere. Carbon dioxide is 50% higher than it was before the Industrial Revolution. Updated April 2024. Snow is melting earlier. Snow is melting earlier in the Northern Hemisphere. Between 1967 and 2022, late spring (April-June) snow cover decreased. Updated August 2022. Show me another fact A complex issue Climate change impacts our society in many different ways. Drought can harm food production and human health. Flooding can lead to spread of disease, death, and damage ecosystems and infrastructure. Human health issues that result from drought, flooding, and other weather conditions increase the death rate, change food availability, and limit how much a worker can get done, and ultimately the productivity of our economy. Hope for the future There is still time to lessen the impacts and severity of climate change. We already know many of the problems and solutionsoffsite link, and researchers continue to find new ones.  Experts believe we can avoid the worst outcomes by reducing emissions to zero as quickly as possible, which will limit warmingoffsite link. To meet this goal, we will have to invest in new technology and infrastructure, which will spur job growth. For example, we will need to continue improving technology and facilities that capture and process renewable energy. Lowering emissions will also benefit human health, saving countless lives and billions of dollars in expenses related to health. Screenshot of Climate Explorer tool interface. A search bar says Explore how climate is projected to change in any county in the United States, and users can enter a city or county. The Climate Explorer: Climate data and projectionsoffsite link Interactive graphs and maps showing how climate conditions in U.S. states and territories are projected to change over the coming decades. Diving deeper into climate change impacts In the following sections, we will look at some of the effects climate change has on our resources and society. Use the links below to jump to a specific section. Water Pyramid Lake is a reservoir near Castaic, California. It stores water for delivery to Los Angeles and other coastal cities of Southern California. It also provides regulated storage for Castaic Powerplant and flood protection along Piru Creek which it dams. Food A close-up photo of a meal that looks professionally made and plated in a shallow bowl with a gold rim and NOAA logo. Human health Houses on the water impacted by erosion. The environment Photo of a sea bass look for food among the schools of smaller fish in this healthy coral reef. Marine biodiversity data and information is critical for understanding the health and status of ecosystems, which is essential for coastal management, conservation and alternative energy planning Infrastructure Aerial image of high tide flooding April 12, 2024, in Annapolis, Maryland. Water The effects of climate change on our water resources can have a big impact on our world and our lives. Patterns of where, when, and how much precipitation falls are changing as temperatures rise. Some areas are experiencing heavier rain events while others are having more droughts. Flooding is an increasing issue as our climate is changing. Compared to the beginning of the 20th century,  precipitation events are stronger, heavier, and more frequent across most of the United States. Drought is also becoming more common, especially in the Western United States. We are using more water during hot weather, especially for agriculture. Much like we sweat more when it is hot out,  hot weather causes plants to lose, or transpire, more water. Then, farmers must give their crops more water. Snowpack is an important source of fresh water for many people. As the snow melts, fresh water becomes available for use. Snowmelt is particularly important in regions like the Western United States where there is not much precipitation in warmer months. But as temperatures warm, there is less snow and snow begins to melt earlier in the year. This means that snowpack is less likely to be a reliable source of water. Food Our food supply depends on climate and weather conditions. Higher temperatures, drought and water stress, diseases, and weather extremes create challenges for farmers and ranchers. Farmers, ranchers, and researchers can address some of these challenges by adapting their methods or creating and using new technology. But, some changes will be difficult to manage, like human and livestock health. Farmworkers can suffer from heat-related health issues, like exhaustion, heatstroke, and heart attacks. Heat can also harm livestock. Is climate change coming for your French fries? Climate and french fries Fries depend on potatoes, and like all crops, potatoes have a preferred climate. How long will America’s favorite side dish have a safe spot on our menu? AnchorHuman health Climate change is already impacting human health. Changes in weather and climate patterns put lives at risk in many ways: Heat is one of the most deadly weather conditions. As ocean temperatures rise, hurricanes are getting stronger and wetter, which can cause death during the hurricane itself and in the aftermath. Dry conditions lead to more wildfires, which bring many health risks. More flooding leads to the spread of waterborne diseases, injuries, death, and chemical hazards. As geographic ranges of mosquitoes, ticks, and other pests expand, they carry diseases to new locations. The environment Climate change affects all living things, or organisms, and the environment they live in, but not equally. The Arctic is one of the ecosystems most vulnerable to the effects of climate change. It is warming at least twice as fast as the global average.  Warming in the Arctic has impacts that are felt across the globe — melting land ice sheets and glaciers contribute dramaticallyoffsite link to sea level rise. Sea levels are also rising due to thermal expansion. Higher sea level puts coastal areas at greater risk of erosion and storm surge. Effects of climate change can build upon one another to damage ecosystems. Sea level rise can cause sediment to smother corals. But, coral reefs are also vulnerable to many other effects of climate change: warming waters can lead to coral bleaching and stronger hurricanes can destroy reefs. Coral reef ecosystems are home to thousands of species, which rely on healthy coral reefs to survive. Some organisms are able to adapt to and even benefit from climate change. Some plants have longer growing seasons or are blooming earlier. But, these changes can happen too fast for other plants and animals to keep up. For example, an earlier blooming plant may depend on a pollinator that does not adapt as quickly. There are also species that have adapted by expanding or shifting their geographic range, meaning they live in new places that used to be too cold or unsuitable in other ways. As a species expands or shifts its range, it may harm other species that already live in the new area. Existing invasive or nuisance species, like lionfish and ticks, may also thrive in even more places because of climate change Ocean ecosystems face an additional challenge: ocean acidification. The ocean absorbs about 30% of the carbon dioxide we release into the atmosphere by burning fossil fuels. As a result, the water is becoming more acidic, which affects marine life. A screenshot of the Exploring Our Changing Ocean storymap. There are selections for Alaska, Mid-Atlantic, Pacific, Gulf, South Atlantic, and North Atlantic. Exploring our changing ocean: Impacts and response to ocean acidification in the U.S.A.offsite link Interactive StoryMaps showing ocean acidification trends, science activities, community engagement, and policy responses taking place in six U.S. regions Infrastructure Physical infrastructure includes bridges, roads, ports, electrical grids, broadband internet, and other parts of our transportation and communication systems. People often design it to be in use for many years. Because of this, most communities have infrastructure that was designed without climate change in mind. Existing infrastructure may not be able to withstand extreme weather events that bring heavy rains, floods, wind, snow, or temperature changes. Impacts that result from these events occur in many different ways. For example, increased temperatures require more indoor cooling, which can put stress on an energy grid. Sudden heavy rainfall that exceeds storm water drainage capacity can lead to flooding that shuts down highways, major transportation routes, and businesses.",
            "Medical Research": "Potatoes, the third most commonly consumed food crop and the main non-cereal food, contribute a substantial quantity of daily energy.1 Several US guidelines classify potatoes as vegetables.23 Although potatoes contain various nutrients such as fiber, vitamin C, potassium, polyphenols, and magnesium, previous research has raised caution about their effects on health.4 The high starch content of potatoes, leading to a high glycemic index and load, combined with possible loss of nutrients and possible health risks resulting from various cooking methods, could contribute to adverse health outcomes.56 Although the health benefits of most vegetables are widely acknowledged, the association between potato consumption and health outcomes, particularly type 2 diabetes (T2D), remains a subject of debate. The results of previous prospective studies investigating the association between potato consumption and T2D were inconclusive. Whereas some studies have indicated a positive association,89 others found no statistically significant association,10111213 and some even observed an inverse association.14 Similar inconsistencies have been found in meta-analyses evaluating these relations.1516 A recent individual participant data (IPD) meta-analysis17 of seven US cohorts found no association with total potato intake but a modest increased risk with fried potato intake. That study did not, however, assess dietary substitutions, which may have an effect on T2D risk. Few studies have estimated the effect on T2D risk of replacing potatoes with alternative carbohydrate sources; though some suggest benefits from replacing potatoes with whole grains and non-starchy vegetables.818 Inconsistent findings may stem from regional differences in potato consumption, variations in methodology (including only baseline versus repeated dietary assessments), inadequate control for confounding variables, and disregard of the potential for reverse causation. Thus, long term observational studies with advanced methodologies and high quality data are needed to clarify these associations. Additionally, substitution analyses are particularly important as the effect of potato consumption might depend on the replacement sources of energy; in contrast with potatoes, whole grains are consistently linked to lower risks of many adverse health outcomes. The extensive data from the Nurses’ Health Study (NHS), Nurses’ Health Study II (NHSII), and Health Professionals Follow-up Study (HPFS), encompassing almost four decades with many repeated assessments of diet and other variables, provide a more precise evaluation of potato and T2D associations than studies with a single, baseline only assessment. Our previous analyses of data from the US health professional cohorts indicated a robust positive association between high potato consumption—particularly French fries—and increased incidence of T2D.18 In the current study, we examined the same three cohorts, including more than 7000 additional people with T2D documented, with extended follow-up, and evaluated the extent to which having only baseline diet assessments, a common limitation in most epidemiological studies, might attenuate associations. Furthermore, we assessed the impact of replacing potatoes with other frequently consumed foods, and explored latency periods along with the possibility of reverse causation. Finally, to inform dietary guidelines, we also conducted an updated dose-response meta-analysis to evaluate the relation between total and specific potato intake and T2D risk and compare the associations of potatoes and whole grains with T2D risk."
        }
        
        selected_sample = st.selectbox("Or choose a sample text:", list(sample_texts.keys()))
        
        if selected_sample != "Select a sample...":
            default_text = sample_texts[selected_sample]
        else:
            default_text = ""
        
        user_text = st.text_area(
            "Enter your text here:",
            value=default_text,
            height=200,
            placeholder="Paste your article, research paper, or any text content here..."
        )
        
        if st.button("🚀 Extract Keywords", type="primary"):
            if user_text.strip():
                with st.spinner("Analyzing text and extracting keywords..."):
                    keywords = get_keywords(user_text, cv, tfidf_trans, feature_names, num_keywords)
                
                if keywords:
                    st.success(f"✅ Successfully extracted {len(keywords)} keywords!")
                    
                    # Store results in session state for persistence
                    st.session_state.keywords = keywords
                    st.session_state.user_text = user_text
                else:
                    st.warning("⚠️ No keywords could be extracted from the provided text. Try with longer or more technical content.")
            else:
                st.error("❌ Please enter some text to analyze.")
    
    with col2:
        st.subheader("💡 Tips")
        st.markdown("""
        **For best results:**
        - Use longer texts (100+ words)
        - Technical or academic content works well
        - Avoid very short or informal texts
        - The model recognizes domain-specific terms
        
        **The algorithm:**
        1. Preprocesses text (remove stopwords, stem, etc.)
        2. Calculates TF-IDF scores
        3. Ranks terms by importance
        4. Returns top keywords
        """)
    
    # Display results if available
    if hasattr(st.session_state, 'keywords') and st.session_state.keywords:
        st.markdown("---")
        st.subheader("📊 Results")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🏷️ Keywords", "☁️ Word Cloud", "📈 Chart"])
        
        with tab1:
            st.subheader("🎯 Extracted Keywords")
            
            # Display keywords in a nice format
            cols = st.columns(2)
            keywords_list = list(st.session_state.keywords.items())
            
            for i, (keyword, score) in enumerate(keywords_list):
                col_idx = i % 2
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="keyword-container">
                        <strong>{keyword}</strong>
                        <span class="score-badge">{score:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Download option
            keywords_df = pd.DataFrame(list(st.session_state.keywords.items()), 
                                     columns=['Keyword', 'TF-IDF Score'])
            csv = keywords_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Keywords as CSV",
                data=csv,
                file_name="extracted_keywords.csv",
                mime="text/csv"
            )
        
        with tab2:
            if show_wordcloud:
                st.subheader("☁️ Word Cloud")
                wordcloud_img = create_wordcloud(st.session_state.keywords)
                if wordcloud_img:
                    st.image(wordcloud_img, use_column_width=True)
            else:
                st.info("Enable 'Show Word Cloud' in the sidebar to view the word cloud.")
        
        with tab3:
            if show_bar_chart:
                st.subheader("📈 Keyword Importance Chart")
                bar_chart = create_bar_chart(st.session_state.keywords)
                if bar_chart:
                    st.plotly_chart(bar_chart, use_container_width=True)
            else:
                st.info("Enable 'Show Bar Chart' in the sidebar to view the chart.")

if __name__ == "__main__":
    main()