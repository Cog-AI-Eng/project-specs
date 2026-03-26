# E-Commerce Purchases Dataset

**File:** `ecommerce_purchases.csv`
**Rows:** 50,000 completed orders
**Users:** ~2,000 unique customers

## Schema

### Identifiers (drop before training)

| Column | Type | Description |
|---|---|---|
| `order_id` | string | Unique order ID (e.g., `ORD-17633387`). |
| `user_id` | string | Customer ID (e.g., `U01600`). |

### Order Features

| Column | Type | Description |
|---|---|---|
| `primary_category` | string | Category of the highest-value product: electronics, clothing, home, sports, books, toys, food, beauty. |
| `num_items` | int | Total quantity of items in the order. |
| `num_distinct_products` | int | Count of unique products in the order. |
| `order_total` | float | Final order total in the order's currency. |
| `payment_method` | string | credit_card, debit_card, paypal, apple_pay, google_pay, bank_transfer. |
| `currency` | string | USD, EUR, GBP, CAD, AUD. |
| `shipping_matches_billing` | int (0/1) | Whether shipping and billing addresses are in the same region. |

### User Behavior Features (7-day window before purchase)

| Column | Type | Description |
|---|---|---|
| `primary_device` | string | Most-used device in the 7 days before purchase: desktop, mobile, tablet. |
| `user_session_count_7d` | int | Login sessions in the 7-day window. |
| `user_page_views_7d` | int | Page views in the 7-day window. |
| `user_cart_adds_7d` | int | Add-to-cart events in the 7-day window. |
| `user_cart_removals_7d` | int | Remove-from-cart events in the 7-day window. |
| `user_searches_7d` | int | Search events in the 7-day window. |

### User History Features

| Column | Type | Description |
|---|---|---|
| `account_age_days` | int | Days since account creation. |
| `previous_completed_purchases` | int | Count of prior completed purchases. |
| `previous_returns` | int | Count of prior returns. |
| `previous_chargebacks` | int (0/1) | Whether this user has any prior chargebacks. |
| `historical_avg_order_value` | float | Average total of prior purchases (0.0 if first-time buyer). |
| `user_return_rate` | float (0.0-1.0) | `previous_returns / previous_completed_purchases` (0.0 if no history). |
| `avg_days_between_purchases` | float | Average gap between purchases (0.0 if fewer than 2 prior purchases). |
| `days_since_last_purchase` | int | Days since this customer's most recent prior purchase (0 if first purchase). |

### Target Columns

| Column | Type | Description |
|---|---|---|
| `customer_ltv_90d` | float | **Regression target.** Predicted customer spend over the next 90 days. You can't look this up at order time -- it hasn't happened yet. |
| `churned_within_60d` | int (0/1) | **Binary classification target.** Did this customer fail to make another purchase within 60 days? ~54% churn rate. |
| `returned_order` | int (0/1) | **Binary classification target.** Was this specific order returned? Predicted at order time, before the return happens. ~35% return rate. |

## Prediction Tasks

Choose **at least one** target. If you train on multiple, each is a separate model. Drop the other target columns to avoid data leakage.

| Task | Target | Type | Agent Use (Project 2) |
|---|---|---|---|
| Customer Lifetime Value | `customer_ltv_90d` | Regression | Prioritize high-value customers, personalize retention offers |
| Churn Prediction | `churned_within_60d` | Binary classification | Flag at-risk customers, trigger retention outreach |
| Return Prediction | `returned_order` | Binary classification | Flag high-risk orders before shipping, preemptive quality checks |

## Categorical Columns Requiring Encoding

- `primary_category` (8 values)
- `payment_method` (6 values)
- `currency` (5 values)
- `primary_device` (3 values)

## XGBoost Data Format

SageMaker's built-in XGBoost expects CSV with **no headers** and the **target column first**. Before uploading to S3, you'll need to:
1. Drop `order_id`, `user_id`, and non-target outcome columns
2. Encode categoricals to numeric (one-hot or ordinal)
3. Move your chosen target to column position 0
4. Save without headers or index
