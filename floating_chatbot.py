import streamlit as st
import google.generativeai as genai
import requests
import os
from datetime import datetime
import json

def init_chatbot():
    """Initialize the chatbot with Google Gemini AI"""
    try:
        # Get API key from environment variable
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            st.error("❌ Google API key not found. Please set GOOGLE_API_KEY environment variable.")
            return None
        
        # Configure Gemini AI
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        return model
    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {str(e)}")
        return None

def get_weather_api_key():
    """Get OpenWeatherMap API key from environment variable"""
    return os.getenv('OPENWEATHER_API_KEY')

def get_real_time_weather(city_name):
    """Get real-time weather data for a city using OpenWeatherMap API"""
    try:
        api_key = get_weather_api_key()
        if not api_key:
            return None, "OpenWeatherMap API key not configured"
        
        # First, get coordinates for the city
        geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},IN&limit=1&appid={api_key}"
        geo_response = requests.get(geocoding_url, timeout=10)
        geo_response.raise_for_status()
        
        geo_data = geo_response.json()
        if not geo_data:
            return None, f"City '{city_name}' not found"
        
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        
        # Get current weather
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        weather_response = requests.get(weather_url, timeout=10)
        weather_response.raise_for_status()
        
        weather_data = weather_response.json()
        
        # Format the weather data
        current_weather = {
            'city': weather_data['name'],
            'country': weather_data['sys']['country'],
            'temperature': round(weather_data['main']['temp'], 1),
            'feels_like': round(weather_data['main']['feels_like'], 1),
            'humidity': weather_data['main']['humidity'],
            'pressure': weather_data['main']['pressure'],
            'description': weather_data['weather'][0]['description'].title(),
            'wind_speed': round(weather_data['wind']['speed'] * 3.6, 1),  # Convert m/s to km/h
            'visibility': weather_data.get('visibility', 'N/A'),
            'sunrise': datetime.fromtimestamp(weather_data['sys']['sunrise']).strftime('%H:%M'),
            'sunset': datetime.fromtimestamp(weather_data['sys']['sunset']).strftime('%H:%M'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return current_weather, None
        
    except requests.exceptions.RequestException as e:
        return None, f"API request failed: {str(e)}"
    except Exception as e:
        return None, f"Error fetching weather data: {str(e)}"

def get_weather_forecast(city_name, days=5):
    """Get weather forecast for a city using OpenWeatherMap API"""
    try:
        api_key = get_weather_api_key()
        if not api_key:
            return None, "OpenWeatherMap API key not configured"
        
        # First, get coordinates for the city
        geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},IN&limit=1&appid={api_key}"
        geo_response = requests.get(geocoding_url, timeout=10)
        geo_response.raise_for_status()
        
        geo_data = geo_response.json()
        if not geo_data:
            return None, f"City '{city_name}' not found"
        
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        
        # Get 5-day forecast
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        forecast_response = requests.get(forecast_url, timeout=10)
        forecast_response.raise_for_status()
        
        forecast_data = forecast_response.json()
        
        # Process forecast data (3-hour intervals for 5 days)
        daily_forecasts = []
        current_date = None
        daily_data = {}
        
        for item in forecast_data['list']:
            date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
            if date != current_date:
                if current_date and daily_data:
                    daily_forecasts.append(daily_data)
                current_date = date
                daily_data = {
                    'date': date,
                    'day': datetime.fromtimestamp(item['dt']).strftime('%A'),
                    'temp_min': item['main']['temp_min'],
                    'temp_max': item['main']['temp_max'],
                    'humidity': item['main']['humidity'],
                    'description': item['weather'][0]['description'].title(),
                    'wind_speed': round(item['wind']['speed'] * 3.6, 1)
                }
            else:
                # Update min/max temperatures
                daily_data['temp_min'] = min(daily_data['temp_min'], item['main']['temp_min'])
                daily_data['temp_max'] = max(daily_data['temp_max'], item['main']['temp_max'])
        
        if daily_data:
            daily_forecasts.append(daily_data)
        
        return daily_forecasts[:days], None
        
    except requests.exceptions.RequestException as e:
        return None, f"API request failed: {str(e)}"
    except Exception as e:
        return None, f"Error fetching forecast data: {str(e)}"

def get_climate_context(current_city=None, current_weather_data=None):
    """Get context about climate and weather for the chatbot"""
    
    base_context = """
    You are ClimatePredict AI Assistant, a helpful AI assistant for weather and climate information.
    
    Your expertise includes:
    - Weather forecasting and analysis
    - Climate change impacts and trends
    - Weather-related safety tips
    - Climate science explanations
    - Indian weather patterns and monsoon
    - Environmental awareness
    
    You can help users with:
    - Understanding weather forecasts
    - Explaining climate phenomena
    - Providing weather safety advice
    - Discussing climate change impacts
    - Answering questions about Indian weather
    - General weather and climate education
    
    IMPORTANT: You are integrated with a weather prediction application that has access to:
    - Machine learning models trained on historical weather data for Indian cities
    - Current weather predictions and forecasts
    - Climate trend analysis
    - Weather pattern recognition
    - REAL-TIME WEATHER DATA from OpenWeatherMap API
    - Live current weather conditions for any city
    - 5-day weather forecasts
    
    When users ask about weather in specific cities (especially Indian cities), you can provide:
    - Information about typical weather patterns for that city
    - Climate characteristics and seasonal variations
    - Weather-related advice and safety tips
    - Explanations of weather phenomena
    
    Always be helpful, accurate, and informative. If you don't know something, say so rather than guessing.
    """
    
    # Add current city context if available
    if current_city:
        base_context += f"\n\nCurrent Context: The user is currently viewing weather information for {current_city}."
    
    # Add current weather data context if available
    if current_weather_data:
        base_context += f"\n\nCurrent Weather Context: {current_weather_data}"
    
    # Add specific guidance for weather questions
    base_context += """
    
    When users ask about weather in specific cities:
    1. If it's an Indian city, provide information about typical weather patterns, climate characteristics, and seasonal variations
    2. Mention that the application has ML models trained on historical data for that city
    3. Provide weather-related advice and safety tips
    4. Explain weather phenomena and climate patterns
    5. If asked about current weather, explain that you can provide insights based on historical patterns and typical conditions
    
    Remember: You are part of a weather prediction application, so you can provide valuable insights about weather patterns and climate information.
    """
    
    return base_context

def render_floating_chatbot(current_city=None, current_weather_data=None):
    """Floating chatbot component for ClimatePredict AI"""
    
    # Initialize session state for chat
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'chatbot_model' not in st.session_state:
        st.session_state.chatbot_model = init_chatbot()
    
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0
    
    # Chatbot UI
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 AI Assistant")
        
        if st.session_state.chatbot_model is None:
            st.error("Chatbot not available. Please check API configuration.")
            return
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")
        
        # Chat input with unique key to prevent loops
        user_input = st.text_input("Ask me about weather or climate:", key=f"chat_input_{st.session_state.chat_input_key}")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            try:
                # Check if user is asking for real-time weather
                weather_keywords = ['weather', 'temperature', 'forecast', 'current', 'now', 'today']
                is_weather_request = any(keyword in user_input.lower() for keyword in weather_keywords)
                
                # Extract city name from user input
                city_name = None
                if is_weather_request:
                    # Try to extract city name from the input
                    words = user_input.lower().split()
                    for word in words:
                        if word in ['in', 'at', 'for']:
                            # Look for city name after these prepositions
                            try:
                                idx = words.index(word)
                                if idx + 1 < len(words):
                                    potential_city = words[idx + 1]
                                    # Handle multi-word cities
                                    if idx + 2 < len(words) and words[idx + 2] not in ['weather', 'temperature', 'forecast', 'current', 'now', 'today']:
                                        potential_city += ' ' + words[idx + 2]
                                    city_name = potential_city.title()
                                    break
                            except ValueError:
                                pass
                    
                    # If no city found, use current city
                    if not city_name and current_city:
                        city_name = current_city
                
                # Get real-time weather data if requested
                weather_info = ""
                if is_weather_request and city_name:
                    with st.spinner("🌤️ Fetching real-time weather data..."):
                        current_weather, error = get_real_time_weather(city_name)
                        if current_weather:
                            weather_info = f"""
REAL-TIME WEATHER DATA for {city_name}:
📍 Location: {current_weather['city']}, {current_weather['country']}
🌡️ Temperature: {current_weather['temperature']}°C (Feels like: {current_weather['feels_like']}°C)
💧 Humidity: {current_weather['humidity']}%
🌬️ Wind Speed: {current_weather['wind_speed']} km/h
☁️ Conditions: {current_weather['description']}
🌅 Sunrise: {current_weather['sunrise']} | 🌇 Sunset: {current_weather['sunset']}
⏰ Last Updated: {current_weather['timestamp']}
"""
                        elif error:
                            weather_info = f"⚠️ Could not fetch real-time weather: {error}"
                
                # Get AI response
                context = get_climate_context(current_city, current_weather_data)
                prompt = f"{context}\n\nUser: {user_input}\n\n{weather_info}\n\nAI Assistant:"
                
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.chatbot_model.generate_content(prompt)
                    ai_response = response.text
                
                # Add AI response to history
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
                # Increment key to clear input and prevent loops
                st.session_state.chat_input_key += 1
                st.rerun()
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.session_state.chat_input_key += 1
                st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.chat_input_key += 1
            st.rerun()
        
        # Help text
        st.markdown("---")
        st.markdown("""
        **💡 Try asking:**
        - "What's the weather in New Delhi?" (Real-time data!)
        - "Current temperature in Mumbai"
        - "Weather forecast for Bangalore"
        - "What causes monsoon rains?"
        - "How does climate change affect weather?"
        - "Weather safety tips for extreme heat"
        """)

if __name__ == "__main__":
    floating_chatbot() 