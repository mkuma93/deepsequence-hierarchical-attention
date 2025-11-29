# Architecture Guide

## Overview

DeepSequence Hierarchical Attention implements a three-level attention mechanism for time series forecasting:

```
Level 1: Feature-Level (TabNet)
    ↓
Level 2: Component-Level (Ensemble)
    ↓
Level 3: Zero Detection (Intermittent Mode)
```

---

## Component Architecture

### 1. Trend Component

**Input Features**: Time-based (day, week, month, year)

**Architecture**:
```
Time Features → TabNet Encoder → Sparse Attention → Dense(32) → Output
     ↓              ↓                  ↓                 ↓
  [4 features]  [Feature Selection]  [Key Features]  [Trend Signal]
```

**Purpose**: Captures long-term patterns and growth trends

---

### 2. Seasonal Component

**Input Features**: Fourier transform features (sin/cos pairs)

**Architecture**:
```
Fourier Features → TabNet Encoder → Sparse Attention → Dense(32) → Output
      ↓                 ↓                  ↓                ↓
  [8-16 features]  [Period Selection]  [Key Periods]  [Seasonal Pattern]
```

**Purpose**: Captures repeating patterns (weekly, monthly)

---

### 3. Holiday Component

**Input Features**: Distance to holidays, holiday flags

**Architecture**:
```
Holiday Features → TabNet Encoder → Sparse Attention → Dense(32) → Output
      ↓                 ↓                  ↓                ↓
  [Multiple holidays] [Holiday Selection] [Key Holidays] [Holiday Effect]
```

**Purpose**: Captures special events and holidays impact

---

### 4. Regressor Component

**Input Features**: Lag features + external variables

**Architecture**:
```
Lag/External Features → TabNet Encoder → Sparse Attention → Dense(32) → Output
         ↓                    ↓                  ↓                ↓
    [Lags + vars]      [Feature Selection]  [Key Features]  [Regression Signal]
```

**Purpose**: Captures autoregressive patterns and external influences

---

## TabNet Encoder Details

Each component uses a TabNet encoder for sequential feature selection:

```python
Input → Attentive Transformer → Feature Transformer → Output
         (Select features)        (Transform selected)
```

**Key Properties**:
- **Sequential Attention**: N_steps of feature selection
- **Sparse Masks**: Produces interpretable feature importance
- **Ghost Batch Normalization**: Stable training with small batches
- **GLU Activations**: Gated Linear Units for controlled information flow

---

## Ensemble Layer

After all components produce outputs, they are combined:

```
Component Outputs → SKU Embedding → Softmax Weights → Weighted Sum
  [4 x hidden]          [8-dim]        [4 weights]     [Base Forecast]
       ↓                   ↓                ↓                 ↓
   Trend=0.3          SKU-specific     sum(w)=1         final_output
   Seasonal=0.5       patterns         per SKU
   Holiday=0.1
   Regressor=0.1
```

**Properties**:
- Different SKUs learn different component importance
- Softmax ensures weights sum to 1
- Low temperature (0.1) for numerical stability

---

## Intermittent Mode Architecture

When `enable_intermittent_handling=True`:

```
Base Forecast → Cross Layers → Zero Probability
                      ↓
                [Feature Interactions]
                      ↓
                Sigmoid(probability)
                      ↓
Final Forecast = Base × (1 - Zero_Prob)
```

**Cross Layers (DCN)**:
- Explicit feature interactions: x₀xᵢ
- Multiple layers for higher-order interactions
- Captures zero patterns without manual engineering

---

## Data Flow Example

### Input Shape
```python
X_features: (batch_size, n_features)  # e.g., (64, 20)
X_sku: (batch_size, 1)                # SKU identifier
```

### Component Processing
```python
# Trend Component
trend_features = X[:, 0:4]           # Extract time features
trend_tabnet = TabNetEncoder(trend_features)  # (64, 8)
trend_attention = SparseAttention(trend_tabnet)  # (64, 8)
trend_output = Dense(32)(trend_attention)  # (64, 32)

# Similar for Seasonal, Holiday, Regressor
```

### Ensemble
```python
# Stack component outputs
components = tf.stack([
    trend_output,      # (64, 32)
    seasonal_output,   # (64, 32)
    holiday_output,    # (64, 32)
    regressor_output   # (64, 32)
], axis=1)  # (64, 4, 32)

# SKU embedding
sku_embed = Embedding(n_skus, 8)(X_sku)  # (64, 8)

# Compute weights
logits = Dense(4)(sku_embed)  # (64, 4)
weights = Softmax(temperature=0.1)(logits)  # (64, 4)

# Weighted sum
base_forecast = sum(components * weights)  # (64, 32)
```

### Intermittent Handler
```python
# Cross layers for feature interactions
cross_out = CrossLayers(n_layers=2)([
    base_forecast,   # (64, 32)
    sku_embed       # (64, 8)
])  # (64, 40)

# Zero probability
zero_prob = Dense(1, activation='sigmoid')(cross_out)  # (64, 1)

# Final forecast
final_forecast = base_forecast * (1 - zero_prob)
```

### Output
```python
{
    'base_forecast': (64, 1),      # Ensemble output
    'zero_probability': (64, 1),   # P(demand=0)
    'final_forecast': (64, 1)      # Final prediction
}
```

---

## Training Details

### Loss Function
```python
# Only train on final_forecast
loss = mae(y_true, y_pred['final_forecast'])
```

### Optimizer
```python
optimizer = Adam(learning_rate=0.001)
```

### Regularization
- Dropout: 0.1 in all components
- L2 on embeddings: Prevents overfitting to SKU IDs
- Ghost Batch Norm: Stabilizes training

---

## Interpretability

### 1. TabNet Feature Importance
Each component provides feature importance scores:
```python
# Extract from TabNet masks
trend_importance = trend_tabnet.feature_mask  # Which time features matter
seasonal_importance = seasonal_tabnet.feature_mask  # Which periods matter
```

### 2. Component Weights
```python
# Per SKU, which components are important
sku_weights = model.get_ensemble_weights(sku_id)
# e.g., {'trend': 0.2, 'seasonal': 0.6, 'holiday': 0.1, 'regressor': 0.1}
```

### 3. Zero Probability
```python
# Why does model predict zero?
zero_prob = predictions['zero_probability']
# High value → Model confident in zero demand
```

---

## Numerical Stability

### Softmax Temperature
```python
weights = softmax(logits / temperature)
# temperature = 0.1 prevents extreme weights
# Avoids NaN from exp(large_number)
```

### Small Epsilon
```python
safe_division = numerator / (denominator + 1e-7)
# Prevents division by zero
```

### Gradient Clipping
```python
# Built into optimizer
optimizer = Adam(clipnorm=1.0)
```

---

## Performance Optimization

### @tf.function Optimization
```python
@tf.function(reduce_retracing=True)
def predict_fn(X_features, X_sku):
    return model([X_features, X_sku], training=False)
```

**Benefits**:
- Compiles to TensorFlow graph
- Reduces Python overhead
- Faster inference (3-5x speedup)

### Batch Processing
```python
# Process multiple SKUs simultaneously
predictions = model.predict([X_batch, sku_batch])  # (batch_size, ...)
```

---

## Extending the Architecture

### Adding New Components

```python
class CustomComponentBuilder:
    def build(self, inputs, sku_embedding):
        # Your custom feature extraction
        features = extract_custom_features(inputs)
        
        # TabNet encoder
        encoded = TabNetEncoder(
            feature_dim=16,
            output_dim=8,
            n_steps=3
        )(features)
        
        # Sparse attention
        attended = SparseAttention(
            hidden_dim=8,
            temperature=0.1
        )(encoded)
        
        # Output layer
        output = Dense(32, activation='relu')(attended)
        return output
```

### Custom Loss Functions

```python
def custom_loss(y_true, y_pred_dict):
    forecast_loss = mae(y_true, y_pred_dict['final_forecast'])
    zero_loss = binary_crossentropy(y_zero, y_pred_dict['zero_probability'])
    return forecast_loss + 0.1 * zero_loss
```

---

## References

- **TabNet**: [Attention-based Neural Networks for Tabular Data](https://arxiv.org/abs/1908.07442)
- **DCN**: [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)
- **Sparse Attention**: [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)
