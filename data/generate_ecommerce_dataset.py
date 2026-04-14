"""
Instructor utility -- regenerate the ecommerce purchases CSV.

Not distributed to students. The CSV itself is provided as-is.

Run:  python generate_ecommerce_dataset.py
Out:  ecommerce_purchases.csv  (50000 orders)

Seed is fixed so every run produces identical output.
"""

import numpy as np
import pandas as pd

SEED = 42
N_ORDERS = 50_000

N_USERS = 2000
N_PRODUCTS = 500
USER_POOL = [f"U{i:05d}" for i in range(N_USERS)]
PRODUCT_POOL = [f"PROD_{1000 + i}" for i in range(N_PRODUCTS)]

PRODUCT_CATEGORIES = ["electronics", "clothing", "home", "sports", "books", "toys", "food", "beauty"]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "apple_pay", "google_pay", "bank_transfer"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "AUD"]
DEVICES = ["desktop", "mobile", "tablet"]


def generate():
    rng = np.random.default_rng(SEED)

    # -- Users with varying activity levels --
    user_popularity = rng.dirichlet(np.ones(N_USERS) * 1.5)
    user_ids = rng.choice(USER_POOL, size=N_ORDERS, p=user_popularity)

    # Per-user stable traits (looked up by index)
    user_idx = np.array([int(u[1:]) for u in user_ids])
    rng_user = np.random.default_rng(SEED + 1)
    user_base_clv = rng_user.lognormal(5.5, 1.0, size=N_USERS)
    user_churn_tendency = rng_user.beta(2, 5, size=N_USERS)
    user_return_tendency = rng_user.beta(2, 8, size=N_USERS)

    # -- Order features --
    primary_category = rng.choice(
        PRODUCT_CATEGORIES, size=N_ORDERS,
        p=[0.22, 0.20, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06],
    )

    cat_price_mu = {"electronics": 4.8, "clothing": 3.5, "home": 3.9, "sports": 3.7,
                    "books": 2.5, "toys": 3.0, "food": 2.3, "beauty": 2.9}
    order_total = np.array([rng.lognormal(cat_price_mu[c], 0.6) for c in primary_category])
    order_total = np.clip(order_total, 5.99, 3000).round(2)

    num_items = rng.choice([1, 1, 1, 2, 2, 3, 4, 5], size=N_ORDERS)
    num_distinct_products = np.minimum(num_items, rng.integers(1, 4, size=N_ORDERS))

    payment_method = rng.choice(PAYMENT_METHODS, size=N_ORDERS,
                                p=[0.35, 0.20, 0.20, 0.10, 0.10, 0.05])
    currency = rng.choice(CURRENCIES, size=N_ORDERS, p=[0.60, 0.15, 0.10, 0.10, 0.05])
    shipping_matches_billing = rng.binomial(1, 0.82, size=N_ORDERS)

    # -- User behavior features (7-day window) --
    user_session_count_7d = rng.poisson(4, size=N_ORDERS).clip(1, 30)
    user_page_views_7d = (user_session_count_7d * rng.poisson(5, size=N_ORDERS)).clip(1, 200)
    user_cart_adds_7d = rng.poisson(3, size=N_ORDERS).clip(0, 25)
    user_cart_removals_7d = (user_cart_adds_7d * rng.binomial(1, 0.35, size=N_ORDERS) *
                             rng.integers(1, 3, size=N_ORDERS)).clip(0, 15)
    user_searches_7d = rng.poisson(2, size=N_ORDERS).clip(0, 20)
    primary_device = rng.choice(DEVICES, size=N_ORDERS, p=[0.45, 0.40, 0.15])

    # -- User history features --
    account_age_days = rng.integers(1, 1500, size=N_ORDERS)
    previous_completed_purchases = rng.poisson(8, size=N_ORDERS).clip(0, 80)
    previous_returns = np.array([
        rng.binomial(previous_completed_purchases[i], user_return_tendency[user_idx[i]])
        if previous_completed_purchases[i] > 0 else 0
        for i in range(N_ORDERS)
    ])
    previous_chargebacks = rng.binomial(1, 0.02, size=N_ORDERS)
    historical_avg_order_value = np.array([
        rng.lognormal(3.5, 0.5) if previous_completed_purchases[i] > 0 else 0.0
        for i in range(N_ORDERS)
    ]).round(2)
    safe_purchases = np.maximum(previous_completed_purchases, 1)
    user_return_rate = np.where(
        previous_completed_purchases > 0,
        np.round(previous_returns / safe_purchases, 3),
        0.0
    )
    avg_days_between_purchases = np.where(
        previous_completed_purchases > 1,
        np.round(account_age_days / safe_purchases, 1),
        0.0
    )
    days_since_last_purchase = np.array([
        rng.integers(0, max(1, int(avg_days_between_purchases[i] * 2.5)))
        if previous_completed_purchases[i] > 0 else 0
        for i in range(N_ORDERS)
    ])

    # ========================================================================
    # TARGET 1: customer_ltv_90d (regression)
    #   Predicted future 90-day spend. Driven by user engagement, history,
    #   purchase frequency, and category preferences. Complex nonlinear
    #   interactions make this hard for simple rules.
    # ========================================================================
    ltv = np.zeros(N_ORDERS)
    for i in range(N_ORDERS):
        base = user_base_clv[user_idx[i]] * 0.15
        if previous_completed_purchases[i] > 5:
            base *= 1 + np.log1p(previous_completed_purchases[i]) * 0.2
        if avg_days_between_purchases[i] > 0 and avg_days_between_purchases[i] < 20:
            base *= 1.8
        elif avg_days_between_purchases[i] > 60:
            base *= 0.4
        if days_since_last_purchase[i] > 90:
            base *= 0.3
        if user_session_count_7d[i] > 6:
            base *= 1.3
        if user_cart_adds_7d[i] > 5:
            base *= 1.2
        if account_age_days[i] < 30:
            base *= 0.5
        elif account_age_days[i] > 365:
            base *= 1.1
        cat = primary_category[i]
        if cat == "electronics":
            base *= 1.4
        elif cat == "food":
            base *= 0.7
        base *= rng.lognormal(0, 0.4)
        ltv[i] = base
    customer_ltv_90d = np.clip(ltv, 0, 5000).round(2)

    # ========================================================================
    # TARGET 2: churned_within_60d (binary)
    #   Will the customer NOT make another purchase within 60 days?
    #   Driven by engagement decay, purchase frequency, days since last
    #   purchase, and user-level churn tendency.
    # ========================================================================
    churn_score = np.zeros(N_ORDERS, dtype=float)
    for i in range(N_ORDERS):
        s = 0.0
        s += user_churn_tendency[user_idx[i]] * 3.0
        if days_since_last_purchase[i] > 45:
            s += 1.5
        elif days_since_last_purchase[i] > 20:
            s += 0.6
        if user_session_count_7d[i] <= 1:
            s += 1.2
        elif user_session_count_7d[i] > 6:
            s -= 0.8
        if user_page_views_7d[i] < 3:
            s += 0.7
        if user_cart_adds_7d[i] == 0:
            s += 0.9
        if previous_completed_purchases[i] < 2:
            s += 0.8
        elif previous_completed_purchases[i] > 15:
            s -= 0.5
        if account_age_days[i] < 60:
            s += 0.4
        if previous_chargebacks[i]:
            s += 0.7
        if primary_device[i] == "mobile":
            s -= 0.2
        s += rng.normal(0, 0.8)
        churn_score[i] = s
    churn_prob = 1 / (1 + np.exp(-churn_score + 1.8))
    churned_within_60d = (rng.random(N_ORDERS) < churn_prob).astype(int)

    # ========================================================================
    # TARGET 3: returned_order (binary)
    #   Will THIS specific order be returned? Predicted at order time before
    #   the return happens. Driven by category, order value, user return
    #   history, browsing hesitation signals, and address mismatch.
    # ========================================================================
    return_score = np.zeros(N_ORDERS, dtype=float)
    for i in range(N_ORDERS):
        s = 0.0
        cat = primary_category[i]
        if cat == "clothing":
            s += 1.5
        elif cat == "electronics":
            s += 0.8
        elif cat == "toys":
            s += 0.5
        elif cat == "food":
            s -= 0.8
        elif cat == "beauty":
            s += 0.3
        if order_total[i] > 200:
            s += 0.4
        if order_total[i] > 500:
            s += 0.3
        s += user_return_tendency[user_idx[i]] * 2.5
        if user_return_rate[i] > 0.25:
            s += 0.8
        elif user_return_rate[i] > 0.10:
            s += 0.3
        if account_age_days[i] < 60:
            s += 0.5
        if not shipping_matches_billing[i]:
            s += 0.4
        if user_cart_removals_7d[i] > 3:
            s += 0.5
        if primary_device[i] == "mobile":
            s += 0.2
        if num_items[i] >= 4:
            s += 0.4
        s += rng.normal(0, 0.7)
        return_score[i] = s
    return_prob = 1 / (1 + np.exp(-return_score + 2.5))
    returned_order = (rng.random(N_ORDERS) < return_prob).astype(int)

    # -- Assemble --
    df = pd.DataFrame({
        "order_id": [f"ORD-{rng.integers(10000000, 99999999)}" for _ in range(N_ORDERS)],
        "user_id": user_ids,
        "primary_category": primary_category,
        "num_items": num_items,
        "num_distinct_products": num_distinct_products,
        "order_total": order_total,
        "payment_method": payment_method,
        "currency": currency,
        "shipping_matches_billing": shipping_matches_billing,
        "primary_device": primary_device,
        "user_session_count_7d": user_session_count_7d,
        "user_page_views_7d": user_page_views_7d,
        "user_cart_adds_7d": user_cart_adds_7d,
        "user_cart_removals_7d": user_cart_removals_7d,
        "user_searches_7d": user_searches_7d,
        "account_age_days": account_age_days,
        "previous_completed_purchases": previous_completed_purchases,
        "previous_returns": previous_returns,
        "previous_chargebacks": previous_chargebacks,
        "historical_avg_order_value": historical_avg_order_value,
        "user_return_rate": user_return_rate,
        "avg_days_between_purchases": avg_days_between_purchases,
        "days_since_last_purchase": days_since_last_purchase,
        "customer_ltv_90d": customer_ltv_90d,
        "churned_within_60d": churned_within_60d,
        "returned_order": returned_order,
    })

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("ecommerce_purchases.csv", index=False)

    print(f"Generated {len(df)} orders")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"\n--- Targets ---")
    print(f"customer_ltv_90d: mean=${df['customer_ltv_90d'].mean():.2f}, "
          f"median=${df['customer_ltv_90d'].median():.2f}, "
          f"max=${df['customer_ltv_90d'].max():.2f}")
    print(f"churned_within_60d: {df['churned_within_60d'].mean():.1%} churn rate")
    print(f"returned_order: {df['returned_order'].mean():.1%} return rate")
    print(f"\nCategory distribution:")
    print(df["primary_category"].value_counts().to_string())
    print(f"\nSample rows:")
    cols = ["order_id", "user_id", "primary_category", "order_total",
            "customer_ltv_90d", "churned_within_60d", "returned_order"]
    print(df[cols].head(10).to_string(index=False))
