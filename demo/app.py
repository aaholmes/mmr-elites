"""
MMR-Elites Interactive Demo

A real-time visualization of Quality-Diversity optimization showing
how MMR-Elites maintains diverse, high-quality archives.

Run with: streamlit run demo/app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from typing import Dict, List, Tuple

# Page config
st.set_page_config(
    page_title="MMR-Elites Demo",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stPlotlyChart {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_arm_task(n_dof: int):
    """Cache the task object."""
    from mmr_elites.tasks.arm import ArmTask
    return ArmTask(n_dof=n_dof, use_highdim_descriptor=True)


def plot_arm_configuration(joints: np.ndarray, target: Tuple[float, float] = (0.8, 0.0)) -> go.Figure:
    """Plot a single arm configuration."""
    n_dof = len(joints)
    link_length = 1.0 / n_dof
    
    # Compute joint positions
    angles = np.cumsum(joints)
    x = np.concatenate([[0], np.cumsum(link_length * np.cos(angles))])
    y = np.concatenate([[0], np.cumsum(link_length * np.sin(angles))])
    
    fig = go.Figure()
    
    # Obstacle
    fig.add_shape(type="rect", x0=0.5, y0=-0.25, x1=0.55, y1=0.25,
                  fillcolor="rgba(255,0,0,0.3)", line=dict(color="red"))
    
    # Arm
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                             line=dict(color='#00d4ff', width=3),
                             marker=dict(size=8, color='#00d4ff'),
                             name='Arm'))
    
    # Target
    fig.add_trace(go.Scatter(x=[target[0]], y=[target[1]], mode='markers',
                             marker=dict(size=20, symbol='star', color='lime'),
                             name='Target'))
    
    # Base
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
                             marker=dict(size=15, color='white'),
                             name='Base'))
    
    fig.update_layout(
        xaxis=dict(range=[-0.5, 1.2], scaleanchor="y"),
        yaxis=dict(range=[-0.8, 0.8]),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def plot_archive_2d(descriptors: np.ndarray, fitness: np.ndarray, title: str) -> go.Figure:
    """Plot archive in 2D (using PCA if high-dimensional)."""
    if descriptors.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(descriptors)
    else:
        coords = descriptors
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=fitness,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Fitness")
        ),
        hovertemplate='Fitness: %{marker.color:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="PC1" if descriptors.shape[1] > 2 else "Dim 1",
        yaxis_title="PC2" if descriptors.shape[1] > 2 else "Dim 2",
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def run_evolution_step(task, archive, fitness, descriptors, selector, batch_size, mutation_sigma):
    """Run one evolution step."""
    n_dof = archive.shape[1]
    
    # Select parents and mutate
    parent_idx = np.random.randint(0, len(archive), batch_size)
    offspring = archive[parent_idx] + np.random.normal(0, mutation_sigma, (batch_size, n_dof))
    offspring = np.clip(offspring, -np.pi, np.pi)
    
    # Evaluate
    off_fit, off_desc = task.evaluate(offspring)
    
    # Pool and select
    pool = np.vstack([archive, offspring])
    pool_fit = np.concatenate([fitness, off_fit])
    pool_desc = np.vstack([descriptors, off_desc])
    
    idx = selector.select(pool_fit, pool_desc)
    
    return pool[idx], pool_fit[idx], pool_desc[idx]


def main():
    st.title("🤖 MMR-Elites: Quality-Diversity Optimization")
    st.markdown("""
    **Watch how MMR-Elites discovers diverse robot arm configurations in real-time.**
    
    The algorithm balances finding high-fitness solutions (reaching the target) with 
    maintaining diversity (exploring different arm postures).
    """)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Parameters")
    
    n_dof = st.sidebar.slider("Degrees of Freedom", 5, 50, 20)
    archive_size = st.sidebar.slider("Archive Size (K)", 100, 2000, 500)
    lambda_val = st.sidebar.slider("Diversity Weight (λ)", 0.0, 1.0, 0.5)
    batch_size = st.sidebar.slider("Batch Size", 50, 500, 200)
    mutation_sigma = st.sidebar.slider("Mutation σ", 0.01, 0.5, 0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **λ Parameter:**
    - λ=0: Pure fitness (greedy)
    - λ=1: Pure diversity (novelty)
    - λ=0.5: Balanced (recommended)
    """)
    
    # Initialize session state
    if 'archive' not in st.session_state or st.sidebar.button("🔄 Reset"):
        try:
            import mmr_elites_rs
            st.session_state.selector = mmr_elites_rs.MMRSelector(archive_size, lambda_val)
            st.session_state.rust_available = True
        except ImportError:
            st.error("Rust backend not available. Run: maturin develop --release")
            st.session_state.rust_available = False
            return
        
        task = get_arm_task(n_dof)
        st.session_state.task = task
        
        # Initialize archive
        archive = np.random.uniform(-np.pi, np.pi, (archive_size, n_dof))
        fitness, descriptors = task.evaluate(archive)
        idx = st.session_state.selector.select(fitness, descriptors)
        
        st.session_state.archive = archive[idx]
        st.session_state.fitness = fitness[idx]
        st.session_state.descriptors = descriptors[idx]
        st.session_state.generation = 0
        st.session_state.history = {'gen': [], 'qd': [], 'max_fit': [], 'mean_fit': []}
    
    # Main display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Archive visualization
        fig = plot_archive_2d(
            st.session_state.descriptors, 
            st.session_state.fitness,
            f"Archive (Gen {st.session_state.generation})"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Best arm configuration
        best_idx = np.argmax(st.session_state.fitness)
        st.subheader("🏆 Best Configuration")
        arm_fig = plot_arm_configuration(st.session_state.archive[best_idx])
        st.plotly_chart(arm_fig, use_container_width=True)
        
        # Metrics
        st.metric("Max Fitness", f"{st.session_state.fitness.max():.4f}")
        st.metric("Mean Fitness", f"{st.session_state.fitness.mean():.4f}")
        st.metric("QD-Score", f"{st.session_state.fitness.sum():.1f}")
    
    # Learning curve
    if st.session_state.history['gen']:
        fig_history = go.Figure()
        fig_history.add_trace(go.Scatter(
            x=st.session_state.history['gen'],
            y=st.session_state.history['qd'],
            mode='lines',
            name='QD-Score'
        ))
        fig_history.update_layout(
            title="Learning Progress",
            xaxis_title="Generation",
            yaxis_title="QD-Score",
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig_history, use_container_width=True)
    
    # Evolution controls
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("▶️ Step (1 gen)"):
            archive, fitness, descriptors = run_evolution_step(
                st.session_state.task,
                st.session_state.archive,
                st.session_state.fitness,
                st.session_state.descriptors,
                st.session_state.selector,
                batch_size,
                mutation_sigma
            )
            st.session_state.archive = archive
            st.session_state.fitness = fitness
            st.session_state.descriptors = descriptors
            st.session_state.generation += 1
            
            # Update history
            st.session_state.history['gen'].append(st.session_state.generation)
            st.session_state.history['qd'].append(fitness.sum())
            st.session_state.history['max_fit'].append(fitness.max())
            st.session_state.history['mean_fit'].append(fitness.mean())
            
            st.rerun()
    
    with col_btn2:
        if st.button("⏩ Run 10 gens"):
            for _ in range(10):
                archive, fitness, descriptors = run_evolution_step(
                    st.session_state.task,
                    st.session_state.archive,
                    st.session_state.fitness,
                    st.session_state.descriptors,
                    st.session_state.selector,
                    batch_size,
                    mutation_sigma
                )
                st.session_state.archive = archive
                st.session_state.fitness = fitness
                st.session_state.descriptors = descriptors
                st.session_state.generation += 1
                
                st.session_state.history['gen'].append(st.session_state.generation)
                st.session_state.history['qd'].append(fitness.sum())
                st.session_state.history['max_fit'].append(fitness.max())
                st.session_state.history['mean_fit'].append(fitness.mean())
            
            st.rerun()
    
    with col_btn3:
        if st.button("🚀 Run 100 gens"):
            progress = st.progress(0)
            for i in range(100):
                archive, fitness, descriptors = run_evolution_step(
                    st.session_state.task,
                    st.session_state.archive,
                    st.session_state.fitness,
                    st.session_state.descriptors,
                    st.session_state.selector,
                    batch_size,
                    mutation_sigma
                )
                st.session_state.archive = archive
                st.session_state.fitness = fitness
                st.session_state.descriptors = descriptors
                st.session_state.generation += 1
                
                if i % 10 == 0:
                    st.session_state.history['gen'].append(st.session_state.generation)
                    st.session_state.history['qd'].append(fitness.sum())
                    st.session_state.history['max_fit'].append(fitness.max())
                    st.session_state.history['mean_fit'].append(fitness.mean())
                
                progress.progress((i + 1) / 100)
            
            st.rerun()


if __name__ == "__main__":
    main()
