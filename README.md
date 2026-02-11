# Motivation

## Model selection

SARIMA and Prophet models were chosen because of their simplicity and narutal ability to work with univariate Time Series data. Also, these models can take into account seasonality while some SKU has yearly seasonal pattern. For the SKU with less then 10 points of data simply constant SKU-level heuristics was chosen as a predict

With less time limitations more complex models are recommended like LSTM. Also, with presence of more data (e.g. numerical: price, stock; categorical: product placement in stores, product type; future: planned price, advertisement features, weather forecast) it's relevant to use tree-based models like LightGBM or more complex sequence-to-sequence Neural Networks like Temporal Fusion Fransformer or LLM-like models like Chronos 2

Also, SKU with insufficient data can be forecasted with slightly more sophisticated heuristics, for example by mean forecast from SKU with close sales

## Confidence Intervals

For SARIMA and Prohet confidene intervals were produced by build-in methods

## Performance Evaluation

I suggest to use a set of metrics and analyzie them depending on customer's needs:

- WAPE - weighted absolute persantage error - main metric
- WAPE max - WAPE with maximum of predict and fact sales in denominator - useful when overpredict is much better then underpredict: for example for products without or with long expiration date, when there is a lot of inventory place, before some event with expected sales growth that can't be predicted by the model (news, political events)
- Bias - error divided by fact sales - useful for overall understanding wether there is under- or overpredict
- three bins of Bias histogram: (-inf, -0.2], (-0.2, 0.2], (0.2, +inf) - useful for understanding how frequently predictions are extremely wrong; also useful for decision making about using another prediction methods on corner cases
- WAPE with absolute error aggregated by SKU - useful for long-term planning

# Inventory management strategy

1. Prediction intervals for different certanity levels can be used as lower and upper bounds of probable demand. If prediction intervals are wide, it can mean that the sales behaves untypical and it's better to apply expert's correction to predict. Also as a shop owner I would use upper interval for inventory holding and not the predict itself, because lost sales and out-of-stock are more harmful then overstock and also besaue we should have some safety stocks
2.


# Instruction for code running

1. Clone this git repo
2. Install all required modules from requirements.txt
3. Execute pipeline from command line: `python3 pipeline.py < path_to_input_data >`. Result will be saved into two files:
  - `predict/predict_week.parquet` - predict to 8 weeks on weekly granularity
  - `predict/predict_month.parquet` - predict to 12 month on monthly granularity

Each file contains the following columns:
  - `date` - date
  - `predict` - sales predict on date
  - `lower_0.2`, `upper_0.2`; `lower_0.5`, `upper_0.5`; `lower_0.8`, `upper_0.8` - 20%, 50% and 80% confidense intervals
