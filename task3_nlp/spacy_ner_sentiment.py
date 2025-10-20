"""
Task 3: NLP with spaCy
Text Data: Amazon Product Reviews
Goal: Named Entity Recognition (NER) and Sentiment Analysis
Author: [Kipruto Andrew Kipngetich]
Date: October 2025
"""

# Import required libraries
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("TASK 3: NLP WITH SPACY - NER AND SENTIMENT ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: LOAD SPACY MODEL
# ============================================================================
print("\n[STEP 1] Loading spaCy Language Model...")

try:
    # Load English language model
    nlp = spacy.load("en_core_web_sm")
    print("✓ spaCy model 'en_core_web_sm' loaded successfully!")
except OSError:
    print("⚠ Model not found. Installing...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("✓ Model installed and loaded!")

print(f"  - Model: {nlp.meta['name']}")
print(f"  - Version: {nlp.meta['version']}")
print(f"  - Language: {nlp.meta['lang']}")

# ============================================================================
# STEP 2: LOAD AND PREPARE SAMPLE DATA
# ============================================================================
print("\n[STEP 2] Loading Sample Amazon Product Reviews...")

# Since we can't download Kaggle datasets directly, we'll create sample data
# In real implementation, you would load from CSV: pd.read_csv('amazon_reviews.csv')

sample_reviews = [
    "I absolutely love my new iPhone 13 Pro! Apple really outdid themselves with this one. The camera is amazing and the battery lasts all day. Highly recommend!",
    "Terrible experience with Samsung Galaxy S21. Screen broke after just 2 weeks. Customer service at BestBuy was unhelpful. Would not buy again.",
    "The Sony WH-1000XM4 headphones are incredible! Best noise cancellation I've ever experienced. Worth every penny. Amazon delivered it quickly too.",
    "Disappointed with my Nike Air Max 270 shoes. They were uncomfortable and fell apart after a month. Nike customer service refused to help.",
    "Fantastic laptop! The Dell XPS 15 is powerful and sleek. Intel Core i7 processor handles everything smoothly. Great purchase from Amazon.",
    "This MacBook Pro from Apple is the best computer I've ever owned. The M1 chip is lightning fast. Love shopping at the Apple Store!",
    "Worst purchase ever. The Microsoft Surface Pro 8 constantly freezes. Microsoft support was useless. Waste of $1000.",
    "Amazing smart watch! The Apple Watch Series 7 tracks everything perfectly. Integration with iPhone is seamless. Love it!",
    "Good value for money. The Google Pixel 6 has an excellent camera. Google Photos integration is convenient. Bought from Target.",
    "Not impressed with Bose QuietComfort 35 II. Sound quality is mediocre. Expected more for $300. Amazon should lower the price.",
    "The Canon EOS R5 camera is a game-changer! Professional quality at a reasonable price. B&H Photo Video had great service.",
    "Horrible tablet. The Amazon Fire HD 10 is slow and laggy. Screen quality is poor. Even for $150 it's not worth it.",
    "Love my new AirPods Pro! Apple's noise cancellation technology is fantastic. Battery life is good. Purchased from Best Buy.",
    "The Nintendo Switch OLED is perfect for gaming on the go. Nintendo always delivers quality. GameStop had it in stock!",
    "Disappointed with Fitbit Versa 3. Inaccurate heart rate monitoring. Fitbit app is buggy. Expected better from Google-owned Fitbit.",
    "Outstanding TV! The LG OLED C1 has incredible picture quality. Dolby Vision looks stunning. Best Buy's Geek Squad set it up perfectly.",
    "Terrible phone. OnePlus 9 Pro overheats constantly. Customer support in China was impossible to reach. Regret buying from Amazon.",
    "The PlayStation 5 is worth the wait! Sony created an amazing gaming experience. Bought mine from Target after months of searching.",
    "Great coffee maker! The Keurig K-Elite makes perfect coffee every time. Sturdy build quality. Amazon Prime delivery was fast.",
    "Awful experience. The HP Pavilion laptop died after 6 months. HP warranty service was a nightmare. Never buying HP again."
]

# Create DataFrame
df = pd.DataFrame({
    'review': sample_reviews,
    'review_id': range(1, len(sample_reviews) + 1)
})

print(f"\n✓ Sample data loaded!")
print(f"  - Total reviews: {len(df)}")
print(f"  - Average review length: {df['review'].str.len().mean():.0f} characters")

print(f"\nSample Reviews:")
print("="*80)
for i in range(3):
    print(f"\nReview {i+1}:")
    print(f"  {df['review'].iloc[i][:100]}...")

# ============================================================================
# STEP 3: NAMED ENTITY RECOGNITION (NER)
# ============================================================================
print("\n" + "="*80)
print("[STEP 3] Performing Named Entity Recognition (NER)")
print("="*80)

# Process all reviews and extract entities
all_entities = []
product_names = []
brands = []
organizations = []
money = []
locations = []

print("\nProcessing reviews for entity extraction...")

for idx, review in enumerate(df['review']):
    # Process review with spaCy
    doc = nlp(review)
    
    # Extract entities
    for ent in doc.ents:
        entity_info = {
            'review_id': idx + 1,
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        }
        all_entities.append(entity_info)
        
        # Categorize entities
        if ent.label_ in ['PRODUCT', 'ORG']:
            product_names.append(ent.text)
        if ent.label_ == 'ORG':
            brands.append(ent.text)
            organizations.append(ent.text)
        if ent.label_ == 'MONEY':
            money.append(ent.text)
        if ent.label_ in ['GPE', 'LOC']:
            locations.append(ent.text)

# Create entities DataFrame
entities_df = pd.DataFrame(all_entities)

print(f"\n✓ Entity extraction completed!")
print(f"  - Total entities found: {len(all_entities)}")
print(f"  - Unique entity types: {entities_df['label'].nunique()}")

# Display entity statistics
print(f"\nEntity Type Distribution:")
entity_counts = entities_df['label'].value_counts()
for entity_type, count in entity_counts.items():
    print(f"  {entity_type}: {count}")

# Display sample entities
print(f"\nSample Extracted Entities:")
print("="*80)
for i in range(min(10, len(entities_df))):
    row = entities_df.iloc[i]
    print(f"  Review {row['review_id']}: '{row['text']}' ({row['label']})")

# Product and Brand Analysis
print(f"\n" + "="*80)
print("PRODUCT AND BRAND ANALYSIS")
print("="*80)

# Most mentioned brands/organizations
brand_counter = Counter(brands)
print(f"\nTop 10 Most Mentioned Brands:")
for brand, count in brand_counter.most_common(10):
    print(f"  {brand}: {count} mentions")

# Product mentions (using ORG entities as proxy for products)
product_counter = Counter(product_names)
print(f"\nTop 10 Most Mentioned Products/Organizations:")
for product, count in product_counter.most_common(10):
    print(f"  {product}: {count} mentions")

# ============================================================================
# STEP 4: RULE-BASED SENTIMENT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[STEP 4] Performing Rule-Based Sentiment Analysis")
print("="*80)

# Define sentiment lexicons
positive_words = {
    'love', 'amazing', 'excellent', 'great', 'fantastic', 'perfect', 'best',
    'incredible', 'outstanding', 'wonderful', 'awesome', 'brilliant', 'superb',
    'good', 'happy', 'impressed', 'recommend', 'worth', 'quality', 'beautiful',
    'powerful', 'fast', 'stunning', 'seamless', 'comfortable', 'convenient',
    'efficient', 'sleek', 'smooth', 'reliable', 'sturdy'
}

negative_words = {
    'terrible', 'awful', 'horrible', 'worst', 'bad', 'poor', 'disappointing',
    'disappointed', 'useless', 'waste', 'broken', 'freezes', 'slow', 'laggy',
    'uncomfortable', 'unhelpful', 'refused', 'regret', 'never', 'not',
    'mediocre', 'inaccurate', 'buggy', 'overheats', 'died', 'nightmare',
    'impossible', 'fell apart'
}

# Intensifiers and negations
intensifiers = {'very', 'extremely', 'absolutely', 'really', 'incredibly', 'highly'}
negations = {'not', 'no', 'never', 'neither', 'nor', 'nothing', 'nowhere', 'hardly'}

def analyze_sentiment(text):
    """
    Rule-based sentiment analysis using lexicons
    Returns: sentiment (positive/negative/neutral) and score
    """
    doc = nlp(text.lower())
    
    positive_score = 0
    negative_score = 0
    
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    for i, token in enumerate(tokens):
        # Check for intensifiers
        intensifier_multiplier = 1.5 if i > 0 and tokens[i-1] in intensifiers else 1.0
        
        # Check for negations (flips sentiment)
        has_negation = i > 0 and tokens[i-1] in negations
        
        if token in positive_words:
            if has_negation:
                negative_score += 1 * intensifier_multiplier
            else:
                positive_score += 1 * intensifier_multiplier
        
        elif token in negative_words:
            if has_negation:
                positive_score += 1 * intensifier_multiplier
            else:
                negative_score += 1 * intensifier_multiplier
    
    # Calculate final sentiment
    total_score = positive_score - negative_score
    
    if total_score > 1:
        sentiment = 'positive'
    elif total_score < -1:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment, total_score, positive_score, negative_score

# Apply sentiment analysis to all reviews
print("\nAnalyzing sentiment for all reviews...")

sentiments = []
for review in df['review']:
    sentiment, score, pos_score, neg_score = analyze_sentiment(review)
    sentiments.append({
        'sentiment': sentiment,
        'score': score,
        'positive_words_count': pos_score,
        'negative_words_count': neg_score
    })

sentiment_df = pd.DataFrame(sentiments)
df = pd.concat([df, sentiment_df], axis=1)

print("\n✓ Sentiment analysis completed!")

# Display sentiment distribution
print(f"\nSentiment Distribution:")
sentiment_counts = df['sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {sentiment.capitalize()}: {count} reviews ({percentage:.1f}%)")

print(f"\nAverage Sentiment Scores:")
print(f"  Mean Score: {df['score'].mean():.2f}")
print(f"  Median Score: {df['score'].median():.2f}")
print(f"  Std Dev: {df['score'].std():.2f}")

# ============================================================================
# STEP 5: DISPLAY DETAILED RESULTS
# ============================================================================
print("\n" + "="*80)
print("[STEP 5] Detailed Analysis Results")
print("="*80)

# Sample positive reviews
print("\n📗 POSITIVE REVIEWS (Sample):")
print("="*80)
positive_reviews = df[df['sentiment'] == 'positive'].head(3)
for idx, row in positive_reviews.iterrows():
    print(f"\nReview {row['review_id']}:")
    print(f"  Text: {row['review'][:150]}...")
    print(f"  Score: {row['score']:.1f} (Positive: {row['positive_words_count']:.0f}, Negative: {row['negative_words_count']:.0f})")
    
    # Extract entities from this review
    doc = nlp(row['review'])
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        print(f"  Entities: {', '.join([f'{e[0]} ({e[1]})' for e in entities[:5]])}")

# Sample negative reviews
print("\n📕 NEGATIVE REVIEWS (Sample):")
print("="*80)
negative_reviews = df[df['sentiment'] == 'negative'].head(3)
for idx, row in negative_reviews.iterrows():
    print(f"\nReview {row['review_id']}:")
    print(f"  Text: {row['review'][:150]}...")
    print(f"  Score: {row['score']:.1f} (Positive: {row['positive_words_count']:.0f}, Negative: {row['negative_words_count']:.0f})")
    
    # Extract entities from this review
    doc = nlp(row['review'])
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        print(f"  Entities: {', '.join([f'{e[0]} ({e[1]})' for e in entities[:5]])}")

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================
print("\n[STEP 6] Creating Visualizations...")

fig = plt.figure(figsize=(20, 12))

# 1. Sentiment Distribution Pie Chart
ax1 = plt.subplot(2, 3, 1)
colors = ['#90EE90', '#FFB6C6', '#ADD8E6']
sentiment_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
ax1.set_ylabel('')

# 2. Top Entity Types
ax2 = plt.subplot(2, 3, 2)
top_entities = entity_counts.head(8)
ax2.barh(range(len(top_entities)), top_entities.values, color='skyblue')
ax2.set_yticks(range(len(top_entities)))
ax2.set_yticklabels(top_entities.index)
ax2.set_xlabel('Count', fontsize=12)
ax2.set_title('Top Entity Types Detected', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

# 3. Top Brands Mentioned
ax3 = plt.subplot(2, 3, 3)
top_brands = dict(brand_counter.most_common(8))
if top_brands:
    ax3.bar(range(len(top_brands)), list(top_brands.values()), color='coral')
    ax3.set_xticks(range(len(top_brands)))
    ax3.set_xticklabels(list(top_brands.keys()), rotation=45, ha='right')
    ax3.set_ylabel('Mentions', fontsize=12)
    ax3.set_title('Top Brands Mentioned', fontsize=14, fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'No brands detected', ha='center', va='center', fontsize=12)
    ax3.set_title('Top Brands Mentioned', fontsize=14, fontweight='bold')

# 4. Sentiment Score Distribution
ax4 = plt.subplot(2, 3, 4)
ax4.hist(df['score'], bins=15, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral')
ax4.set_xlabel('Sentiment Score', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Sentiment Score Distribution', fontsize=14, fontweight='bold')
ax4.legend()

# 5. Positive vs Negative Word Counts
ax5 = plt.subplot(2, 3, 5)
categories = ['Positive\nWords', 'Negative\nWords']
values = [df['positive_words_count'].sum(), df['negative_words_count'].sum()]
bars = ax5.bar(categories, values, color=['green', 'red'], alpha=0.7)
ax5.set_ylabel('Total Count', fontsize=12)
ax5.set_title('Positive vs Negative Word Counts', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 6. Entity Label Word Cloud Style (Top 15)
ax6 = plt.subplot(2, 3, 6)
if len(entities_df) > 0:
    entity_text = ' '.join(entities_df['text'].tolist())
    word_freq = Counter(entity_text.split())
    top_words = dict(word_freq.most_common(15))
    
    if top_words:
        ax6.barh(range(len(top_words)), list(top_words.values()), color='teal')
        ax6.set_yticks(range(len(top_words)))
        ax6.set_yticklabels(list(top_words.keys()))
        ax6.set_xlabel('Frequency', fontsize=12)
        ax6.set_title('Most Frequent Entity Words', fontsize=14, fontweight='bold')
        ax6.invert_yaxis()

plt.tight_layout()
plt.savefig('spacy_nlp_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'spacy_nlp_results.png'")
plt.show()

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================
print("\n[STEP 7] Saving Results...")

# Save results to CSV
df.to_csv('amazon_reviews_analyzed.csv', index=False)
print("✓ Review analysis saved to 'amazon_reviews_analyzed.csv'")

entities_df.to_csv('extracted_entities.csv', index=False)
print("✓ Extracted entities saved to 'extracted_entities.csv'")

# Create summary report
summary = {
    'total_reviews': len(df),
    'positive_reviews': len(df[df['sentiment'] == 'positive']),
    'negative_reviews': len(df[df['sentiment'] == 'negative']),
    'neutral_reviews': len(df[df['sentiment'] == 'neutral']),
    'total_entities': len(entities_df),
    'unique_brands': len(brand_counter),
    'average_sentiment_score': df['score'].mean()
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('sentiment_summary.csv', index=False)
print("✓ Summary statistics saved to 'sentiment_summary.csv'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TASK 3 COMPLETION SUMMARY")
print("="*80)

print("\n✓ All NLP tasks completed successfully!")

print(f"\nNamed Entity Recognition Results:")
print(f"  - Total entities extracted: {len(entities_df)}")
print(f"  - Entity types found: {entities_df['label'].nunique()}")
print(f"  - Top entity type: {entity_counts.index[0]} ({entity_counts.values[0]} occurrences)")
print(f"  - Unique brands identified: {len(brand_counter)}")
print(f"  - Most mentioned brand: {brand_counter.most_common(1)[0][0] if brand_counter else 'None'}")

print(f"\nSentiment Analysis Results:")
print(f"  - Positive reviews: {len(df[df['sentiment'] == 'positive'])} ({len(df[df['sentiment'] == 'positive'])/len(df)*100:.1f}%)")
print(f"  - Negative reviews: {len(df[df['sentiment'] == 'negative'])} ({len(df[df['sentiment'] == 'negative'])/len(df)*100:.1f}%)")
print(f"  - Neutral reviews: {len(df[df['sentiment'] == 'neutral'])} ({len(df[df['sentiment'] == 'neutral'])/len(df)*100:.1f}%)")
print(f"  - Average sentiment score: {df['score'].mean():.2f}")
print(f"  - Overall sentiment: {'Positive ✓' if df['score'].mean() > 0 else 'Negative ✗'}")

print(f"\nFiles Generated:")
print(f"  1. spacy_nlp_results.png (visualizations)")
print(f"  2. amazon_reviews_analyzed.csv (full analysis)")
print(f"  3. extracted_entities.csv (all entities)")
print(f"  4. sentiment_summary.csv (summary statistics)")

print("\n" + "="*80)
print("Task 3 execution complete! 🎉")
print("="*80)

# ============================================================================
# BONUS: EXAMPLE CODE FOR USING SPACY
# ============================================================================
print("\n" + "="*80)
print("BONUS: HOW TO USE SPACY FOR YOUR OWN TEXT")
print("="*80)

print("\nExample code:")
print("""
import spacy

# Load model
nlp = spacy.load("en_core_web_sm")

# Process your text
text = "Apple Inc. released the iPhone 14 in September 2022 for $799."
doc = nlp(text)

# Extract entities
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")
    
# Output:
# Apple Inc. - ORG
# iPhone 14 - PRODUCT
# September 2022 - DATE
# $799 - MONEY

# Analyze tokens
for token in doc:
    print(f"{token.text}: {token.pos_} (Lemma: {token.lemma_})")
""")