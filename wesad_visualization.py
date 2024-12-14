import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import column
from bokeh.resources import INLINE


os.makedirs('visualizations', exist_ok=True)

def load_wesad_data(subject_id):
    """Load data from pickle file for a given subject"""
    try:
        file_path = os.path.join(os.getcwd(), 'data', f'S{subject_id}', f'S{subject_id}.pkl')
        print(f"Loading data from: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data
    except Exception as e:
        print(f"Error loading data for subject {subject_id}: {str(e)}")
        return None

def create_plotly_visualization(data):
    """Create time series visualization using Plotly"""
    sampling_rate = 100
    chest_data = data['signal']['chest']
    
    signal_length = len(chest_data['ECG'])
    time = np.arange(0, signal_length, sampling_rate) / 700
    
    fig = go.Figure()
    colors = {'ECG': 'blue', 'EDA': 'red', 'EMG': 'green'}
    
    for signal_name in ['ECG', 'EDA', 'EMG']:
        signal_data = chest_data[signal_name].reshape(-1)[::sampling_rate]
        
        fig.add_trace(go.Scatter(
            x=time,
            y=signal_data,
            name=signal_name,
            line=dict(color=colors[signal_name])
        ))
    
    fig.update_layout(
        title='Physiological Signals Time Series',
        xaxis_title='Time (seconds)',
        yaxis_title='Signal Amplitude',
        hovermode='x unified',
        width=1200,
        height=600
    )
    
    
    fig.write_html("visualizations/plotly_timeseries.html")
    return fig

def create_seaborn_visualization(data):
    """Create distribution plots using Seaborn"""
    chest_data = data['signal']['chest']
    sampling_rate = 100
    
    df = pd.DataFrame({
        'ECG': chest_data['ECG'].reshape(-1)[::sampling_rate],
        'EDA': chest_data['EDA'].reshape(-1)[::sampling_rate],
        'EMG': chest_data['EMG'].reshape(-1)[::sampling_rate]
    })
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i, signal in enumerate(['ECG', 'EDA', 'EMG']):
        sns.violinplot(data=df[signal], ax=axes[i])
        axes[i].set_title(f'{signal} Signal Distribution')
        axes[i].set_ylabel('Amplitude')
    
    plt.tight_layout()
    
    
    plt.savefig('visualizations/seaborn_distributions.png')
    return fig

def create_altair_visualization(data):
    """Create state distribution visualization using Altair"""
    labels = pd.DataFrame({
        'Label': data['label'],
        'Count': 1
    })
    
    label_counts = labels.groupby('Label').count().reset_index()
    
    condition_map = {
        0: 'Not Defined',
        1: 'Baseline',
        2: 'Stress',
        3: 'Amusement',
        4: 'Meditation'
    }
    label_counts['Condition'] = label_counts['Label'].map(condition_map)
    
    chart = alt.Chart(label_counts).mark_bar().encode(
        x=alt.X('Condition:N', title='Condition'),
        y=alt.Y('Count:Q', title='Number of Samples'),
        color=alt.Color('Condition:N', legend=None),
        tooltip=['Condition', 'Count']
    ).properties(
        title='Distribution of Psychological States',
        width=600,
        height=400
    )
    
    
    chart.save('visualizations/altair_states.html')
    return chart

def create_bokeh_visualization(data):
    """Create signal comparison visualization using Bokeh"""
    sampling_rate = 1000
    chest_data = data['signal']['chest']
    time = np.arange(0, len(chest_data['ECG']), sampling_rate) / 700
    
    p = figure(width=1200, height=600, title='Physiological Signals Comparison')
    
    p.line(time, chest_data['ECG'].reshape(-1)[::sampling_rate], 
           line_color='blue', legend_label='ECG')
    p.line(time, chest_data['EDA'].reshape(-1)[::sampling_rate], 
           line_color='red', legend_label='EDA')
    p.line(time, chest_data['EMG'].reshape(-1)[::sampling_rate], 
           line_color='green', legend_label='EMG')
    
    p.xaxis.axis_label = 'Time (seconds)'
    p.yaxis.axis_label = 'Signal Amplitude'
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    
    
    output_file("visualizations/bokeh_comparison.html")
    save(p)
    return p

if __name__ == "__main__":
    data = load_wesad_data(2)
    
    if data is not None:
        try:
            print("\nCreating visualizations...")
            
            print("Creating Plotly visualization...")
            plotly_fig = create_plotly_visualization(data)
            print("Saved as visualizations/plotly_timeseries.html")
            
            print("Creating Seaborn visualization...")
            seaborn_fig = create_seaborn_visualization(data)
            print("Saved as visualizations/seaborn_distributions.png")
            
            print("Creating Altair visualization...")
            altair_chart = create_altair_visualization(data)
            print("Saved as visualizations/altair_states.html")
            
            print("Creating Bokeh visualization...")
            bokeh_plot = create_bokeh_visualization(data)
            print("Saved as visualizations/bokeh_comparison.html")
            
            print("\nAll visualizations have been saved in the 'visualizations' directory.")
            
        except Exception as e:
            print(f"\nError creating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Failed to load data. Please check the file path and data format.")