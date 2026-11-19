-- late_rate_by_state
SELECT customer_state, COUNT(*) n, ROUND(100.0*AVG(is_late),1) late_pct, ROUND(AVG(delivery_days),1) avg_days
                           FROM orders GROUP BY customer_state HAVING n>=300 ORDER BY late_pct DESC LIMIT 10;

-- revenue_by_category
SELECT category, COUNT(*) orders, ROUND(SUM(price)) revenue, ROUND(AVG(review_score),2) avg_review
                           FROM orders GROUP BY category ORDER BY revenue DESC LIMIT 10;

-- monthly_trend
SELECT purchase_month, COUNT(*) orders, ROUND(SUM(payment_value)) revenue, ROUND(100.0*AVG(is_late),1) late_pct
                      FROM orders WHERE purchase_month BETWEEN '2017-01' AND '2018-08' GROUP BY purchase_month ORDER BY purchase_month;

-- review_by_lateness
SELECT is_late, COUNT(*) n, ROUND(AVG(review_score),2) avg_review,
                           ROUND(100.0*AVG(review_score<=2),1) pct_negative FROM orders GROUP BY is_late;