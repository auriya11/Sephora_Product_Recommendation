# Project Report

## Project Overview

# Skincare Recommendation System for Sephora

## Project Background

The beauty industry, particularly skincare, has experienced significant growth over the last decade. According to *Grand View Research* (2023), the global skincare market is projected to reach **USD 189.3 billion by 2025**, driven by increasing awareness of self-care and skin health.

In this ecosystem, e-commerce platforms like **Sephora** play a vital role in connecting brands with consumers. However, one major challenge in selling skincare products online is **helping users find the right product** for their specific skin needs. Skincare is highly personal, influenced by factors such as:

- Skin type (dry, oily, combination, sensitive)
- Preference for certain active ingredients
- Reviews and experiences from users with similar characteristics

Hence, an **intelligent and contextual recommendation system** is essential to provide accurate, relevant, and personalized product suggestions.

This project utilizes the Sephora Skincare Reviews dataset from Kaggle, which includes over 1 million user reviews with rich features such as:

- User profiles (skin type, skin tone, eye color, hair color)
- Reviews (text, rating, usefulness)
- Product metadata (brand, ingredients, price, category, recommendation label)

---

## Project Objectives

1. **Analyze user satisfaction** based on reviews, ratings, and the `is_recommended` feature.
2. **Identify patterns in user preferences** based on characteristics like skin type.
3. **Build a recommendation system** (content-based, collaborative, and hybrid) that delivers relevant skincare suggestions tailored to individual preferences.

---

## Business Problem Statements

1. **What do users think about skincare products on Sephora, and how satisfied are they?**  
   With over 10,000 reviews, understanding satisfaction levels is key.

2. **Is there a pattern between user characteristics and their preferred products?**  
   For example, do people with sensitive skin rate certain products higher?

3. **How can we build a recommendation system that is personal and relevant to each user’s needs?**  
   Relevant recommendations can improve customer satisfaction and boost conversions.

---

## Business Impact

This project helps:

- Users avoid trial-and-error by finding suitable products faster
- Enhance **user experience** through personalized shopping
- Reduce product return rates and increase brand loyalty
- Provide **business insights** into market segmentation and consumer preference

---

## Solution Approach

### Solution 1: EDA & Feature Engineering

- Clean missing values, remove duplicates, and fix data types
- Extract sentiment from `review_text` and `review_title`
- Normalize categorical features like `skin_type`, `eye_color`, etc.

### Solution 2: Recommendation System

- **Content-Based Filtering**:  
  Uses product attributes (brand, ingredients, price, category, sentiment) and cosine similarity

- **Collaborative Filtering (Item-Based)**:  
  Based on user-product interaction matrix (`is_recommended`, ratings)

- **Hybrid Recommendation**:  
  Combines the strength of content and collaborative methods using a weighted score

---

## Evaluation Metrics

- **Precision@K**: Proportion of top-K recommendations that are relevant
- **Recall@K**: Proportion of relevant items successfully recommended
- **F1-Score@K**: Harmonic mean of precision and recall
- **MAP@K**: Average precision at rank K
- **MRR**: Rank of the first relevant item in the recommendation list

### Model Performance Summary

| Model                    | Precision@5 | Recall@5 | F1@5   | MAP@5  | MRR    |
|--------------------------|-------------|----------|--------|--------|--------|
| Content-Based Filtering  | **0.4000**  | **1.0000** | **0.5714** | 0.5583 | 0.6667 |
| Collaborative Filtering  | 0.2000      | 0.5000   | 0.2857 | 0.5000 | 0.5000 |
| Hybrid Recommendation    | **0.4000**  | **1.0000** | **0.5714** | **0.6417** | **0.7500** |

### Key Insights

- Hybrid Recommendation Model delivers the most **balanced and effective** performance across all metrics
- Hybrid is **more robust**, combining user behavior and product content
- Collaborative Filtering underperforms due to sparse user interaction
- Content-Based model works well, but suffers from **over-specialization**

---

## Summary of Achievements

| Goal                                                                 | Outcome                                                                                   |
|----------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Analyze user satisfaction via reviews and ratings                   | Sentiment analysis aligned with user ratings; identified satisfaction trends              |
| Identify preference patterns based on user traits                   | Discovered strong relations between `skin_type` and product preference                    |
| Develop personal and relevant recommendation system                 | Hybrid model outperforms others in evaluation, delivering the most relevant suggestions   |

---

## Impactful Solutions

| Solution Component                                    | Visible Impact                                                                              |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------|
| **EDA & Feature Engineering**                         | Cleaned dataset and created user/product-level insights like `review_sentiment`, `skin_type` |
| **Content-Based Filtering**                           | Recommended products with similar ingredients or purposes                                  |
| **Collaborative Filtering**                           | Captured implicit preferences across users, though limited by data sparsity                |
| **Hybrid Recommendation**                             | Achieved best performance by combining CBF and CF                                          |
| **Evaluation using Precision, Recall, MAP, MRR, F1**  | Verified that recommendations are not only relevant but also well-ranked                  |

---

## References

- Grand View Research (2023). *Skincare Market Size, Share & Trends Analysis Report*.  
  [https://www.grandviewresearch.com/industry-analysis/skincare-market](https://www.grandviewresearch.com/industry-analysis/skincare-market)

- McKinsey & Company (2021). *The State of Fashion: Beauty*.  
  [https://www.mckinsey.com/industries/retail/our-insights/the-state-of-fashion-beauty](https://www.mckinsey.com/industries/retail/our-insights/the-state-of-fashion-beauty)

---


## Data Understanding

### Dataset Description

This project uses the **Sephora Products and Skincare Reviews** dataset, publicly available on [Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews). It contains **over 1 million entries** detailing user reviews, product information, and user characteristics for both skincare and cosmetic products.

The dataset is rich in features — with **45 columns** — and includes product metadata, user profiles, behavioral data, and review insights. These are essential for understanding user satisfaction and building personalized recommendation systems.

---

### Key Variables Overview

| Column                  | Description                                                                  |
|-------------------------|-------------------------------------------------------------------------------|
| `author_id`             | Unique user identifier                                                       |
| `rating_x`              | User rating for a product                                                    |
| `is_recommended`        | Whether the user recommends the product (1 = yes, 0 = no)                    |
| `review_text`           | Full text of the user review                                                 |
| `review_title`          | Title or summary of the review                                               |
| `skin_type`, `skin_tone`, `eye_color`, `hair_color` | User profile attributes (important for personalization)       |
| `product_id`, `product_name`, `brand_name` | Basic product identifiers                                  |
| `price_usd`, `value_price_usd`, `sale_price_usd` | Price-related features                                  |
| `loves_count`           | How many users "loved" the product                                           |
| `ingredients`           | Product formulation and ingredient list                                      |
| `primary_category`, `secondary_category`, `tertiary_category` | Product categorization              |
| `highlights`            | Key product features (e.g. vegan, cruelty-free)                              |
| `submission_time`       | Timestamp when the review was submitted                                      |
| `helpfulness`           | Review helpfulness score (voted by other users)                              |

---

### Dataset Snapshot

- **Total entries:** 10,000 (sample used in project)
- **Total columns:** 45
- **Mixed data types** across user attributes, numeric ratings, text reviews, and categorical variables.

#### Data Type Breakdown

| Data Type | Columns Count | Example Columns                  |
|-----------|---------------|----------------------------------|
| `int64`   | 13            | `rating_x`, `brand_id`, `child_count`     |
| `float64` | 10            | `price_usd`, `reviews`, `is_recommended` |
| `object`  | 22            | `review_text`, `skin_type`, `product_name` |

---

### Missing Values Overview

The table below shows features with the highest percentage of missing values:

| Column               | Missing Count | Percentage |
|----------------------|----------------|------------|
| `variation_desc`     | 9,920          | 99.2%      |
| `sale_price_usd`     | 9,917          | 99.2%      |
| `value_price_usd`    | 9,730          | 97.3%      |
| `child_min_price`    | 5,876          | 58.8%      |
| `helpfulness`        | 5,216          | 52.2%      |
| `review_title`       | 2,854          | 28.5%      |
| `hair_color`         | 2,033          | 20.3%      |
| `eye_color`          | 1,909          | 19.1%      |
| `skin_tone`          | 1,534          | 15.3%      |
| `is_recommended`     | 1,529          | 15.3%      |
| `skin_type`          | 1,012          | 10.1%      |
| `variation_value`    |   628          | 6.3%       |
| `size`               |   402          | 4.0%       |
| `ingredients`        |   201          | 2.0%       |
| `review_text`        |    16          | 0.2%       |

> **Note:**  
> Critical columns such as `skin_type`, `review_text`, and `is_recommended` are important for analysis and the recommendation engine. Missing values in these fields must be handled carefully.

---

### Why This Dataset?

The rich combination of **user attributes, product features, and review texts** makes this dataset highly suitable for building a **context-aware recommender system** that aligns with skincare needs, user preferences, and behavioral insights.

---

# Exploratory Data Analysis (EDA)

## 1. Rating & Recommendation Distribution

![Rating Distribution](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/download%20(20).png?raw=true)

- Most users gave a **5-star rating**, with over **6,000 reviews** falling in this category.
- Low ratings (1–2) are rare, suggesting a generally **satisfied user base**.

**Insight:**  
Users tend to leave reviews when they have a **very positive experience** with the product.

---

## 2. Skin Type Preferences

### a. Dominant Skin Type

![Skin Type Distribution](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/download%20(21).png?raw=true)

- The majority of users have **combination skin**, nearing **5,000 users**.

### b. Other Skin Types
- **Dry:** ~1,700 users  
- **Normal:** ~1,300 users  
- **Oily:** ~1,100 users  

**Insight:**  
Recommendation systems should strongly **prioritize combination skin users**, the largest segment in the dataset.

---

## 3. Text Feature Analysis

### a. Wordcloud Review

![Wordcloud](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/download%20(22).png?raw=true)

- Most common words: **skin**, **product**, **love**, **use**, **face**
- Positive tones like: **love**, **great**, **amazing**

### b. Top 20 Most Common Words

![Top 20 Words](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/download%20(23).png?raw=true)

**Insight:**  
User reviews are generally **positive**, focused on **skincare benefits**, and provide rich descriptive feedback.

---

## 4. Product Price Distribution

![Price Distribution](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/download%20(24).png?raw=true)

- Most products are priced **under $100**, with a peak between **$30–$50**.
- The price distribution is **right-skewed**, indicating only a few premium-priced products.

**Insight:**  
Affordable products dominate the market, offering a **strategic recommendation point** for most users.

---

## 5. Most Reviewed Products

![Top Reviewed Products](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/download%20(25).png?raw=true)

- **"Lip Sleeping Mask"** is the most reviewed product — showing strong interest in **hydrating lip care**.
- Product variety: **cleanser**, **serum**, **mask**, indicating multi-stage skincare demand.
- Focused keywords in top products: **hydration**, **gentle**, **antioxidant**.
- "Mini" versions are highly reviewed — **sample-sized items** are popular.
- Top brands include: **Laneige**, **Fresh**, **The Ordinary**, **Dr. Dennis Gross**, and **Youth To The People**.

**Insight:**  
**Brand familiarity**, sample options, and hydration benefits drive high review activity.

---

## 6. Average Product Ratings

![Average Product Ratings](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/Screenshot%202025-04-29%20154026.png?raw=true)

- Many top-reviewed products maintain an **average rating of 5.0**, reinforcing the trend of **positive sentiment**.

---

## 7. Sentiment Analysis

![Sentiment Distribution](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/download%20(26).png?raw=true)

Out of 10,000 review samples:

- **Positive:** 7,826 reviews  
- **Neutral:** 1,766 reviews  
- **Negative:** 408 reviews  

**Insight:**  
The review landscape is dominated by **positive sentiment**, supporting the trend seen in rating distribution and text analysis.

---

# Data Preparation

This stage focuses on cleaning, formatting, and engineering the dataset to ensure high-quality inputs for analysis and modeling.

---

## 1. Duplicate Removal

- **Method:** `drop_duplicates()`  
- **Why:** Duplicated rows can bias model training and skew data distribution.

---

## 2. Dropping High-Missing Columns

- **Removed Columns:**  
  `helpfulness`, `review_title`, `hair_color`, `variation_desc`, `value_price_usd`, `sale_price_usd`, `child_max_price`, `child_min_price`

- **Why:**  
  These columns have **>20% missing values** and contribute minimal useful information for the recommendation system.

---

## 3. Handling Missing Values

### a. Mode Imputation for Categorical Columns

- **Columns:**  
  `skin_type`, `skin_tone`, `eye_color`, `is_recommended`, `variation_type`, `variation_value`, `size`

- **Why:**  
  These user and product features are essential for user profiling and demographic-based recommendations.

### b. Empty String Imputation

- **Column:** `review_text`  
- **Why:** Retains rows for NLP sentiment analysis, even if the review is empty.

### c. Custom Value Imputation

| Column              | Imputed With |
|---------------------|--------------|
| `ingredients`       | `"Unknown"`  |
| `tertiary_category` | `"Other"`    |

- **Why:** Preserves categorical integrity for products with partial metadata.

---

## 4. Data Type Transformation

- **Column:** `submission_time` → converted to `datetime` format  
- **Why:** Enables temporal trend analysis and time-based user behavior segmentation.

---

## 5. Feature Engineering

### a. Product Deduplication

- **Action:** Remove repeated `product_name_x` entries  
- **Why:** Prevent recommendation bias caused by identical product entries.

### b. Combined Feature Construction

- **Features Merged Into Text:**  
  `brand_name_x`, `highlights`, `ingredients`, `primary_category`

- **New Column:** `combined_features`  
- **Why:** Creates a full product profile for content-based recommendation.

### c. TF-IDF Vectorization

- **Method:** `TfidfVectorizer`  
- **Why:** Converts `combined_features` into a weighted numeric matrix that highlights rare but important keywords like `"niacinamide"` or `"retinol"`.

---

## 6. Product Similarity Computation

- **Method:** Cosine Similarity  
- **Output:** Product-to-product similarity matrix

- **Why:** Enables content-based filtering by identifying the **most similar products** based on ingredients, brand, and features.

---

# Modeling

This project implements **three recommendation system models** to generate personalized and relevant **Top-10 skincare product suggestions**:

---

## 1. Content-Based Filtering (TF-IDF + Cosine Similarity)

### Approach  
This model builds feature vectors for each product based on:

- Brand  
- Highlights  
- Ingredients  
- Category  

These features are combined into a new field (`combined_features`), vectorized using **TF-IDF**, and compared using **cosine similarity** to recommend similar products.

---

### Example Recommendation

**Input Product:**  
*Water Bank Blue Hyaluronic Cream Moisturizer* - LANEIGE (Skincare)

**Top-10 Recommendations:**

| Product Name                                          | Brand     | Category |
|-------------------------------------------------------|-----------|----------|
| Water Bank Blue Hyaluronic Hydration Set             | LANEIGE   | Skincare |
| Green Tea Hyaluronic Acid Hydrating Moisturizer      | innisfree | Skincare |
| Water Bank Blue Hyaluronic Serum                     | LANEIGE   | Skincare |
| Water Bank Blue Hyaluronic Eye Cream                 | LANEIGE   | Skincare |
| Retinol Firming Cream Treatment                      | LANEIGE   | Skincare |
| Ultra Facial Advanced Repair Barrier Cream           | Kiehl’s   | Skincare |
| Green Tea Hyaluronic Acid Hydrating Serum            | innisfree | Skincare |
| Water Sleeping Mask with Squalane                    | LANEIGE   | Skincare |
| Water Bank Blue Hyaluronic Gel Moisturizer           | LANEIGE   | Skincare |
| Clarifying Cleansing Foam with Bija Seed Oil         | innisfree | Skincare |

---

### Insight

- The system captures **brand consistency** and **product line similarity** (e.g., Water Bank series).
- Cross-brand recommendations like innisfree and Kiehl’s are also suggested due to similar product purposes (hydrating, soothing, etc.).

---

## 2. Collaborative Filtering (Item-Based, Cosine Similarity)

### Approach  
Uses a **user-item interaction matrix** (based on `is_recommended`, `rating`) and calculates product similarity based on **historical user behavior** (item-based).

---

### Example Recommendation

**Input Product:**  
*Mini Unseen Sunscreen SPF 40* - Supergoop! (Skincare)

**Top-10 Recommendations:**

| Product Name                                               | Brand              | Category |
|------------------------------------------------------------|--------------------|----------|
| GOOPGLOW Glow Lotion                                       | goop               | Skincare |
| Vinoperfect Brightening Glycolic Peel Mask                 | Caudalie           | Skincare |
| FILLING GOOD Hyaluronic Acid Plumping Serum                | Farmacy            | Skincare |
| The Rich Cream with TFC8 Face Moisturizer                  | Augustinus Bader   | Skincare |
| Clarifique Exfoliating Hydrating Face Essence              | Lancôme            | Skincare |
| Squalane + BHA Pore-Minimizing Toner                       | Biossance          | Skincare |
| Avocado Soothing Skin Barrier Serum                        | Glow Recipe        | Skincare |
| Resurfacing Overnight Peel with Retinol & Niacinamide      | Kate Somerville    | Skincare |
| CLOUD JELLY Plumping Hydration Serum                       | Herbivore          | Skincare |
| Super Anti-Aging Face Cream                                | Dr. Barbara Sturm  | Skincare |

---

### Insight

- Recommends relevant skincare products from **different brands** based on **similar user preferences**.
- Encourages users to **explore new brands** while staying within their skincare needs.

---

## 3. Hybrid Recommendation (Content-Based + Collaborative Filtering)

### Approach  
Combines both content-based and collaborative filtering using a weighted formula:

Hybrid Score = α * content_score + (1 - α) * collaborative_score

In this project, **α = 0.6** (favoring content-based).


### Example Recommendation

**Input Product:**  
*Mini The Rich Cream with TFC8 Face Moisturizer* - Augustinus Bader

**Top-10 Recommendations:**

| Product Name                                          | Brand                   | Category |
|-------------------------------------------------------|--------------------------|----------|
| The Eye Cream with TFC8                              | Augustinus Bader         | Skincare |
| The Cream Cleansing Gel                              | Augustinus Bader         | Skincare |
| The Ultimate Soothing Cream                          | Augustinus Bader         | Skincare |
| Mini The Cream with TFC8                             | Augustinus Bader         | Skincare |
| The Cream with TFC8                                  | Augustinus Bader         | Skincare |
| The Light Cream                                       | Augustinus Bader         | Skincare |
| The Face Cream Mask                                   | Augustinus Bader         | Skincare |
| Advanced Retinol + Ferulic Wrinkle Cream             | Dr. Dennis Gross         | Skincare |
| Avocado Nourishing Hydration Mask                    | Kiehl’s                  | Skincare |
| Clearly Clean Makeup Removing Cleansing Balm         | Farmacy                  | Skincare |

---

### Insight

- Dominated by **same-brand recommendations** (Augustinus Bader) for consistency and brand loyalty.
- Also includes **cross-brand suggestions** that share similar **functionality** (hydrating, anti-aging, soothing).
- Combines **content similarity** and **user preferences** for highly personalized suggestions.

---

## Summary of Models

| Model                  | Strengths                                                                 |
|------------------------|---------------------------------------------------------------------------|
| Content-Based Filtering| Suggests similar products in terms of content and formulation. Ideal for loyal users or exploring within a product line. |
| Collaborative Filtering| Recommends products based on collective user behavior. Great for discovering new brands or trending products. |
| Hybrid Recommendation  | Best of both worlds — personalized, contextual, and highly relevant.     |

---

# Evaluation

To assess the performance of the skincare product recommendation system, we used evaluation metrics specifically suited for top-k recommendation tasks. These metrics measure how relevant and accurate the recommended items are to the users.

---

## Evaluation Metrics Used

1. **Precision@k**  
   ![precision](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/Screenshot%202025-04-30%20111905.png?raw=true)  
   Measures how many of the top-k recommended items are actually relevant.

2. **Recall@k**  
   ![recall](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/Screenshot%202025-04-30%20111933.png?raw=true)  
   Measures the proportion of relevant items successfully retrieved in the top-k list.

3. **F1-Score@k**  
   ![f1score](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/Screenshot%202025-04-30%20111959.png?raw=true)  
   Harmonic mean of Precision and Recall — reflects the balance between both.

4. **MAP@k (Mean Average Precision)**  
   ![MAP](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/Screenshot%202025-04-30%20112038.png?raw=true)  
   Averages the precision scores at each relevant item retrieved, considering ranking order.

5. **MRR (Mean Reciprocal Rank)**  
   ![MRR](https://raw.githubusercontent.com/auriya11/recommendationsystem/main/Screenshot%202025-04-30%20112109.png?raw=true)  
   Measures how early the first relevant item appears in the recommendation list.

---

## Model Evaluation Results

| Model                    | Precision@5 | Recall@5 | F1@5   | MAP@5  | MRR    |
|--------------------------|-------------|----------|--------|--------|--------|
| Content-Based Filtering  | **0.4000**  | **1.0000** | **0.5714** | 0.5583 | 0.6667 |
| Collaborative Filtering  | **0.2000**  | **0.5000** | **0.2857** | **0.5000** | **0.5000** |
| Hybrid Recommendation    | **0.4000**  | **1.0000** | **0.5714** | **0.6417** | **0.7500** |

---

## Insight & Analysis

### Content-Based Filtering

- Achieved **high precision and recall**, meaning the recommended products are very relevant based on product features.
- However, **MAP and MRR** are slightly lower than the hybrid model, indicating that although relevant, items might not always appear early in the ranked list.

### Collaborative Filtering

- Lower **precision and recall**, as this model relies heavily on user interaction data, which may be sparse or incomplete.
- **Performs worse in cold-start or low-data scenarios**, where fewer interactions are available.

### Hybrid Recommendation

- **Highest overall performance across all metrics**.
- **Best MRR**, meaning relevant products appear earlier in the ranked list.
- **Best MAP**, showing better prioritization in recommendation ranking.
- Successfully **combines strengths of both content and interaction-based approaches**.

---

## Problem Statement Review & Business Goal Alignment

### 1. What is the level of user satisfaction toward Sephora's skincare products?

**Approach:**  
Analyzed `rating`, `is_recommended`, and performed sentiment analysis on `review_text`.

**Findings:**  
- Most users gave high ratings (4–5).
- `is_recommended` aligns well with positive sentiment.
- Sentiment analysis confirmed that positive reviews correlate with higher ratings.

---

### 2. Is there a relationship between user characteristics (skin_type, eye_color, hair_color) and their product preferences?

**Approach:**  
Segmented users by attributes such as `skin_type`, then analyzed preferred products.

**Findings:**  
- Certain skin types (e.g., oily, combination) are more inclined to prefer specific products.
- This opens the door for **personalized recommendations** based on user profiles.

---

### 3. How can we build a personalized and relevant skincare recommendation system?

**Approach:**  
Developed and evaluated three models:

- **Content-Based Filtering (CBF):** Based on product attributes.
- **Collaborative Filtering (CF):** Based on historical user-product interactions.
- **Hybrid Model:** Combines CBF and CF with weighted average scoring.

**Result:**  
The **Hybrid Model** outperformed others, with:
- **MRR:** 0.75  
- **MAP@5:** 0.64  
- **Recall@5:** 1.0  

---

## Project Goals & Achievements

| Goal                                                                 | Achievements                                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Analyze user satisfaction based on reviews and ratings              | EDA and sentiment analysis confirmed high user satisfaction and alignment with product ratings.   |
| Identify patterns between user characteristics and product choices  | Found meaningful links between `skin_type` and product preference, enabling potential segmentation. |
| Build a personalized and relevant recommendation engine             | Hybrid model achieved the highest scores across all evaluation metrics.                           |

---

## Solution Breakdown & Impact

| Solution                               | Impact                                                                                      |
|----------------------------------------|---------------------------------------------------------------------------------------------|
| **EDA & Feature Engineering**          | Cleaned the dataset and engineered features such as `skin_type`, `review_sentiment`, etc.   |
| **Content-Based Filtering**            | Suggested products similar in brand, ingredients, and category.                             |
| **Collaborative Filtering**            | Discovered related products based on behavior of similar users.                             |
| **Hybrid Recommendation System**       | Merged the benefits of CBF and CF; achieved top metrics in MRR, MAP, and overall accuracy.  |
| **Evaluation using multiple metrics**  | Ensured that recommendations are not only relevant but also well-ranked and diverse.        |

---

## Final Conclusion

This recommendation system successfully addresses Sephora's business needs:

- Users are generally satisfied with skincare products (proven via ratings and sentiment).
- `skin_type` and other personal attributes are crucial for **personalized product matching**.
- The **Hybrid Recommendation Model** offers the best balance between relevance and diversity.

### Impact for Sephora:

- Increases **user engagement and loyalty**
- Delivers **more targeted product suggestions**
- Offers **data-driven insights** for brands regarding market segmentation and user preference

---

