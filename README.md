## Motivation

### Model selection

Prophet was chosen as main models because of its simplicity and narutal ability to work with univariate Time Series data. Also, Prophet takes into account seasonality while some SKU clearly has yearly seasonal pattern. For the SKU with less then 10 points of data simply SKU-level heuristics was chosen as a predict: mean sales of last year of data.

Choise between SARIMA and Prophet was done by scoring on cross-validation on a year of data. The best model on monthly level was heuristics, it can mean that either models needs more tuning or another models should be used. Nevertheless, Prophet was chosen because it has better aggregation WAPE metrics while has Bias > 0, also it predicts sales picks well.

### Confidence Intervals

For Prohet confidene intervals were produced by build-in methods. Under the hood it uses Monte-Carlo sampling from population of predicts, than confidence intervals are comouted as quantiles from this samples.

For heuristics confidence intervals are computed as quantiles of residuals on leave-one-out validation. Because the model is constant, we can't use methods based on assumption of residual normality

### Performance Evaluation

I suggest to use a set of metrics and analyzie them depending on customer's needs:

- WAPE - weighted absolute persantage error - main metric
- WAPE max - WAPE with maximum of predict and fact sales in denominator - useful when overpredict is much better then underpredict: for example for products without or with long expiration date, when there is a lot of inventory place, before some event with expected sales growth that can't be predicted by the model (news, political events)
- Bias - error divided by fact sales - useful for overall understanding wether there is under- or overpredict
- three bins of Bias histogram: (-inf, -0.2], (-0.2, 0.2], (0.2, +inf) - useful for understanding how frequently predictions are extremely wrong; also useful for decision making about using another prediction methods on corner cases
- WAPE with absolute error aggregated by SKU - useful for long-term planning

## Inventory management strategy

1. Prediction intervals for different certanity levels can be used as lower and upper bounds of probable demand. If prediction intervals are wide, it can mean that prediction horizon is too far and too many factors can affect it. Or, if the invervals are wide on close horizons, it means that the demand behaves untypical and it's better to take into considerations another forecast techniques like expert's estimation. On the other side, if the real sales are close to upper prediction interval, it can be interpreted as unexpected demand growth and we should procude enough products to cover it. For example, if predict for yesterday is 10, upper bounds of 80% and 50% intervals are 12 and 15 respectfully and fact sales are 12, we should treat as predict upper bounds of 50% interval.

Also as a shop owner I would use upper interval for inventory holding and not the predict itself, because lost sales and out-of-stock are more harmful for the business then overstock. Also it's better to produce according to upper bound because we should have safety stocks in case of any incidents on supply chain.

2. I think the best plan is to take predictions in month granularity (ensured that our model takes into consideration year seasonality and holidays), take predict to next moth, then take upper bound of predict interval with 90% of certanity. It means that with 90% certanity sales will not be more then this number, it means that we will cover all sales without taking too much space

3. For the next quarter I would use the same strategy: take three monthly predicts, then sum up upper bounds of predict intervals for 90% certanity


## Instruction for code running

1. Clone this git repo
2. Install all required modules from requirements.txt
3. Execute pipeline from command line: `python3 pipeline.py < path_to_input_data >`. Result will be saved into two files:
  - `predict/predict_week.parquet` - predict to 8 weeks on weekly granularity
  - `predict/predict_month.parquet` - predict to 12 month on monthly granularity

For the test results of inventory management open and run jupyter notebook `Quarter_inventory_analysus.ipynb`

Each file contains the following columns:
  - `date` - date
  - `predict` - sales predict on date
  - `lower_0.2`, `upper_0.2`; `lower_0.5`, `upper_0.5`; `lower_0.8`, `upper_0.8` - 20%, 50% and 80% confidense intervals


## Further improvements

Some things that would increase predict quality with less strict deadlines:

- With less time limitations more complex models are recommended like LSTM. Also, with presence of more data (e.g. numerical: price, stock; categorical: product placement in stores, product type; future: planned price, advertisement features, weather forecast) it's relevant to use tree-based models like LightGBM or more complex sequence-to-sequence Neural Networks like Temporal Fusion Fransformer or LLM-like models like Chronos 2.
- All of these models (except Prophet) would benefit from adding holiday and weekend markup
- Predicts of simplier models like Prophet and ARIMA can be used as features in more complex models, also sometimes it's useful to model seasonality by sin/cos series decomposition.
- Also, SKU with insufficient data can be forecasted with slightly more sophisticated heuristics, for example by mean forecast of SKU with close sales or, in general, of SKU from the same cluster
- For SARIMA models, hyperparameters can be chosen differently for each SKU cluster
- Unit and end-to-end tests should be added to ensure reproducibility
