# subsidence

1. create_mask
2. process_raster
3. cleanup
4. create_zero_subsidence_data
5. assign_subsidence
6. fit_pipeline
7. train_model
8. map_prediction





(a)—Locations of recorded subsidence rates. Includes 34 sources with reported subsidence maps that were subsequently digitized and converted to points at
30″ resolution, 193 sources with a single subsidence point per source, and 19221 GPS stations. (b)—Frequency distribution of the subsidence rates. (c–e)—Digitization
workflow. (c)—Points detected by the digitization algorithm, based on the provided colors in the legend of the original image in Figure S1 of the Supporting
Information S1. (d)—Visualization of the extracted points in GIS software, based on extracted longitude, latitude, and rates. (e)—Final locations of the points, after
resampling and snapping on the 30″ tiles of the predictor maps.





Global prediction of land subsidence, with relevant feature importance and zonal statistics at the top of the figure.
Modeled subsidence rates for the entire globe (a), zoomed-in maps of land subsidence for North America (b), South America
(c), Europe and North Africa (d), Middle East (e), and South, East, and South-East Asia (f). The zoomed in maps were
visualized with unique color scales that are different from the global map (a) and are included in Supporting Information S1
with additional maps of land subsidence at global, continental, and regional level in Figures S8–S14 of the Supporting
Information S1. The model was trained using the subsidence data points presented in Figure 1 and a comprehensive set of
environmental parameters listed in Table S2 of the Supporting Information S1. Gaussian smoothing has been applied to
enhance the visualization (including Figures S8–S14 in Supporting Information S1).





Summary of the methodology and results. Simplified architecture of utilized machine learning model (a),
proportions of land area and population affected worldwide by land subsidence larger than 5 mm/year by region (b), and
importance and impact of the predictors (c).