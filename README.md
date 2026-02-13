## Motivation

### Model selection

I chose Prophet as a main model because of its simplicity and natural ability to work with univariate Time Series data. Also, Prophet takes into account seasonality while some SKU clearly has yearly seasonal pattern. For the SKU with fewer than 10 points of data, a simple SKU-level heuristic was chosen as a predictor: mean sales of the last year of sales.

Also, I tried to use SARIMA models. Final decision was made based on scoring models' predictions on cross-validation that took a year of historical data. The best models on the monthly level were Prophet and the heuristic, it can mean that either the models need more tuning or another model should be used. Nevertheless, Prophet was chosen because it has better SKU-level aggregation WAPE metrics while having Bias > 0, and it predicts sales picks well.

### Confidence Intervals

For Prophet, confidence intervals were produced by built-in methods. Under the hood, Prophet uses Monte-Carlo sampling from a population of predictions, then confidence intervals are computed as quantiles from these samples.

For heuristic, confidence intervals were computed as quantiles of residuals on leave-one-out validation. We can't use methods based on the assumption of residual normality because the model is constant

### Performance Evaluation

I suggest using a set of metrics and analyzing them depending on stakeholders' needs:

- WAPE - weighted absolute percentage error - main metric
- WAPE max - WAPE with maximum of predict and fact sales in denominator - useful when overpredict is much better than underpredict: for example, for products without or with long expiration date, when there is a lot of inventory place, before some event with expected sales growth that can't be predicted by the model (news, political events)
- Bias - error divided by fact sales - useful for overall understanding whether there is under- or overprediction
- three bins of Bias histogram: (-inf, -0.2], (-0.2, 0.2], (0.2, +inf) - useful for understanding how frequently predictions are extremely wrong; also useful for decision making about using another prediction method on corner cases
- WAPE with absolute error aggregated by SKU-level - useful for long-term planning

## Inventory management strategy

1. Prediction intervals for different certainty levels can be used as lower and upper bounds of probable demand. If prediction intervals are wide, it can mean that the prediction horizon is too far and too many factors can affect it. Or, if the invervals are wide on close horizons, it means that the demand behaves untypically and it's better to take into consideration another forecast techniques like an expert's estimation. On the other side, if the real sales are close to the upper prediction interval, it can be interpreted as unexpected demand growth, and we should produce enough products to cover it. For example, if the prediction for yesterday is 10, the upper bounds of 80% and 50% CI intervals are 12 and 15 respectfully and fact sales are 12, we should take as predict upper bounds of 50% CI interval from the future forecasts.

Also, as a shop owner, I would use the upper CI interval for inventory holding, because lost sales and out-of-stock are more harmful for the business than overstock. Also, it's better to produce a little more then mean demand estimation because we must have safety stocks in case of any incidents on the supply chain.

1. In my opinion, the best plan is to take predictions in month granularity (ensuring that our model takes into consideration year seasonality and holidays), take predict to next month, then take the upper bound of the prediction interval with 90% of certainty. It means that with 90% certanity, sales will not be more than this number, it means that we will cover all sales without taking too much space

2. For the next quarter, I would use the same strategy: take three monthly predictions, then sum up the upper bounds of the prediction intervals for 90% CI.

3. The method described in pt. 2 produces an estimation = 223 898 072 quantity. From metrics on historical data, we can assume that all of this amount will be bought, but probably we will need to supply more product by the end of the quarter to meet customers' demand

## Instruction for code running

1. Clone this git repo
2. Install all required modules from requirements.txt
3. Go to `src` directory
4. Execute pipeline from command line: `python3 pipeline.py < path_to_input_data >`. The result will be saved into two files:
  - `predict/predict_week.parquet` - predict to 8 weeks on weekly granularity
  - `predict/predict_month.parquet` - predict to 12 months on monthly granularity

For the test results of inventory management, install all required modules from requirements-dev.txt, then open and run jupyter notebook `Quarter_inventory_analysus.ipynb`

Each file contains the following columns:

  - `date` - date
  - `predict` - sales predict
  - `lower_0.2`, `upper_0.2`; `lower_0.5`, `upper_0.5`; `lower_0.8`, `upper_0.8` - 20%, 50% and 80% confidense intervals

## Further improvements

Some things would increase the quality of the predictions and the code, provided there were less strict deadlines:

Model performance:

- I would use more complex models like LSTM. Also, with the presence of more data (e.g., numerical: price, stock; categorical: product placement in stores, product type; future: planned price, advertisement features, weather forecast), it's relevant to use tree-based models like LightGBM or more complex sequence-to-sequence Neural Networks like Temporal Fusion Transformer or LLM-like models like Chronos 2.
- All of these models (except Prophet) would benefit from adding holiday and weekend markup
- Simpler models' predictions (models like Prophet and ARIMA) can be used as features in more complex models. Also, sometimes it's useful to model seasonality by sin/cos series decomposition.
- Also, SKUs with insufficient data can be forecasted with slightly more sophisticated heuristics, for example, by the mean forecast of SKUs with close sales or, in general, of SKUs from the same cluster
- For SARIMA models, hyperparameters can be chosen differently for each SKU cluster
- For more complex models: look at learning curve, feature importance, SHAP

Code:

- Add unit tests to most of functions, add end-to-end tests to train and predict pipelines

Usability in production:

- Save trained models
- Save data and model versions after training new model. Detect latest stable model and use it for prediction.
- Detect if data for predict is new or it was already handled
- Save statistincs on train and predict data and predictions (for example, mean and std in moving window). Alert if new data differs from old data significantly
- Also, for more compelx models, alert if train and validation losses differs from "usual" levels
