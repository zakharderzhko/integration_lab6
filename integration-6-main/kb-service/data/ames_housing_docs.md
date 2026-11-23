# Ames Housing Valuation Reference Documentation

## sec-1-overview | page: 1
The Ames Housing dataset describes residential property sales in Ames, Iowa. Property prices depend primarily on above-grade living area, overall material and finish quality, basement area, garage capacity, year of construction, and municipal zoning classification.

## sec-2-living-area | page: 2
Above-grade living area (Gr Liv Area, measured in square feet) is among the strongest positive predictors of residential sale price. Homes with above-grade living area exceeding 1600 sq ft typically command above-median valuations, while properties above 2000 sq ft sit in the upper price segment.

## sec-3-overall-quality | page: 3
Overall material and finish quality (Overall Qual) rates a house on a scale from 1 (very poor) to 10 (very excellent). Ratings of 7 or higher indicate good to superior construction craftsmanship and are strongly associated with higher sale prices. Ratings of 5 or below correspond to average or dated finishes and lower market valuations.

## sec-4-basement-garage | page: 4
Total basement area (Total Bsmt SF) adds substantial functional space and increases valuation, particularly when basement area exceeds 900 sq ft. Garage area (Garage Area) also contributes positively: garages of 450 sq ft or larger generally accommodate two vehicles and raise buyer demand in suburban neighbourhoods.

## sec-5-year-built | page: 5
Year Built reflects the original construction date and strongly correlates with modern building standards, energy efficiency, and reduced maintenance costs. Properties built after 2000 generally achieve premium valuations compared with houses constructed prior to 1980, even when living areas are similar.

## sec-6-zoning | page: 6
Municipal zoning classification (MS Zoning) defines permitted land use. Residential Low Density (RL) is the most common suburban zoning category and typically commands higher valuations than Residential Medium Density (RM) or Commercial/Floating Village zones due to larger lot sizes and quieter streets.

## sec-7-valuation-policy | page: 7
Document-grounded explanations provide domain context from reference documentation and market guidelines. They do not expose internal tree splits or causal model weights; the final numerical valuation is produced exclusively by the production ML model registered under the @prod alias in MLflow.