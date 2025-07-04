# Urban Heat Island Analysis Project

The Urban Heat Island Analysis project is a initiative focused on developing AI-powered tools for urban climate analysis and planning. The project involves the development of a machine learning protocol that allows urban planners and environmental researchers to predict surface temperatures across different areas of a city based on land use patterns and real-time weather conditions. This project addresses the growing need for data-driven approaches to urban heat management in the context of climate change and sustainable city development.

## Project Information

**University:** VUB

**Research group:** ETRO

**Lead researcher(s):** Andrei Covaci

**Priority domain:** Climate and energy

**SDG:**
- 3. Ensure healthy lives and promote well-being for all at all ages.
- 11. Sustainable Cities and Communities
- 13. Climate Action


## Technology

The project is built upon a design of computer vision algorithms and machine learning models to automatically analyze aerial or satellite imagery and predict surface temperature distributions across urban areas. The system uses ArUco marker detection for precise image rectification, followed by advanced image processing to classify land use types (water bodies, green spaces, and impervious surfaces like concrete and asphalt). 

Using a trained Random Forest model, the technology combines land use analysis with real-time weather data from the Brussels Mobility Twin API to predict surface temperatures across a grid. The system can process different weather scenarios including summer day conditions, summer night conditions, and real-time weather data, providing temperature predictions in a matter of seconds.

Think of it like how weather apps predict temperature, but instead of giving you one temperature for an entire city, this system creates a detailed temperature map showing how different neighborhoods and land types will heat up differently based on their characteristics - whether they have more parks, concrete, or water features.

## Implementation of the Research

The project is currently at the proof-of-concept stage, with a functional backend API that can process images and generate temperature predictions in real-time. The current system can analyze rectified aerial imagery and produce detailed heat matrices showing temperature variations across urban areas within seconds of image upload.

The technology processes images through several sophisticated steps:
1. **Image Rectification:** Uses ArUco markers to correct perspective and ensure accurate geographical mapping
2. **Land Use Classification:** Analyzes RGB pixel data using HSV color space to identify water, vegetation, and impervious surfaces
3. **Spatial Analysis:** Applies convolutional kernels to calculate land use percentages in 250m and 1km radius surrounding each point
4. **Temperature Prediction:** Uses machine learning to predict surface temperatures based on land use composition and weather conditions

The next development phase involves expanding the system to handle larger geographical areas and integrating with existing urban planning tools and GIS systems used by municipal authorities.

## Demonstrator

The demo is a web-based application that enables users to upload aerial imagery and receive instant temperature analysis. Users can upload images containing ArUco markers (representing a geographical area) and select different weather scenarios to see how surface temperatures would vary across that area.

The system automatically:
- Detects and rectifies the uploaded image using ArUco markers
- Analyzes land use patterns within the defined area
- Applies weather conditions (summer day, summer night, or real-time Brussels weather)
- Generates a detailed temperature prediction matrix
- Returns both the processed image and temperature data

The demonstrator offers a practical application of AI for urban climate analysis that goes beyond abstract concepts. It shows how machine learning can be applied to real urban planning challenges, helping city planners understand the thermal implications of different land use decisions. This technology could be instrumental for Brussels and other cities in developing heat mitigation strategies, planning green infrastructure, and adapting to climate change impacts.

By visualizing how different urban configurations affect local temperatures, the demonstrator helps users understand the urban heat island effect and the importance of green spaces, water features, and sustainable urban design in creating more livable cities. 