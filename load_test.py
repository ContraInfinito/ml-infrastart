"""
load_test.py - Concurrent load tester for FastAPI prediction service.

Measures latency (P50/P95/P99) and throughput under concurrent load.
Generates realistic football player feature vectors for regression predictions.

Usage:
    source .venv/bin/activate
    python load_test.py [--host 0.0.0.0] [--port 8000] [--requests 500] [--concurrency 10]
"""

import asyncio
import aiohttp
import numpy as np
import argparse
import time
from datetime import datetime
from typing import List, Dict
import json


def generate_player_features() -> Dict:
    """Generate random realistic football player features for prediction."""
    positions = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
    leagues = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
    specific_positions = [
        "Centre-Forward", "Left Winger", "Right Winger",  # Forward
        "Central Midfielder", "Left Midfielder", "Right Midfielder", "Attacking Midfielder",  # Midfielder
        "Centre-Back", "Left-Back", "Right-Back", "Defensive Midfielder",  # Defender
        "Goalkeeper"  # Goalkeeper
    ]
    trajectories = ["rising_star", "growing", "stable", "declining", "falling_sharply"]
    
    return {
        "age": np.random.uniform(18, 38),
        "career_span_years": np.random.uniform(1, 20),
        "years_to_peak": np.random.uniform(0.5, 15),
        "value_cagr": np.random.uniform(-0.5, 3.0),
        "value_to_peak_cagr": np.random.uniform(0.0, 3.5),
        "value_multiplier_x": np.random.uniform(1, 500),
        "post_peak_decline_pct": np.random.uniform(0, 100),
        "value_volatility": np.random.uniform(0.2, 2.0),
        "mean_yoy_growth_rate": np.random.uniform(-1.0, 2.0),
        "num_valuation_points": np.random.randint(5, 30),
        "num_clubs_career": np.random.randint(1, 10),
        "position_group": np.random.choice(positions),
        "league_name": np.random.choice(leagues),
        "position": np.random.choice(specific_positions),
        "trajectory": np.random.choice(trajectories)
    }


async def send_prediction_request(
    session: aiohttp.ClientSession,
    url: str,
    player_features: Dict
) -> Dict:
    """
    Send a single prediction request and measure latency.
    
    Returns: {
        'latency_ms': float,
        'status': int,
        'success': bool,
        'value_eur': float or None,
        'error': str or None
    }
    """
    payload = {"features": player_features}
    
    try:
        start_time = time.time()
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            latency_ms = (time.time() - start_time) * 1000
            
            if resp.status == 200:
                data = await resp.json()
                return {
                    'latency_ms': latency_ms,
                    'status': resp.status,
                    'success': True,
                    'value_eur': data.get('predicted_value_eur'),
                    'error': None
                }
            else:
                return {
                    'latency_ms': latency_ms,
                    'status': resp.status,
                    'success': False,
                    'value_eur': None,
                    'error': f"HTTP {resp.status}"
                }
    except asyncio.TimeoutError:
        return {
            'latency_ms': None,
            'status': None,
            'success': False,
            'value_eur': None,
            'error': "Request timeout"
        }
    except Exception as e:
        return {
            'latency_ms': None,
            'status': None,
            'success': False,
            'value_eur': None,
            'error': str(e)
        }


async def run_load_test(
    host: str,
    port: int,
    num_requests: int,
    concurrency: int
) -> Dict:
    """
    Run concurrent load test against prediction endpoint.
    
    Args:
        host: API host
        port: API port
        num_requests: Total number of requests to send
        concurrency: Number of concurrent requests
        
    Returns:
        Dictionary with latency statistics and results
    """
    url = f"http://{host}:{port}/predict"
    
    print(f"\nLoad Test Configuration")
    print("=" * 80)
    print(f"Target: {url}")
    print(f"Total requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print("=" * 80)
    
    # Generate player features for all requests
    requests_data = [generate_player_features() for _ in range(num_requests)]
    
    # Run load test
    results = []
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        # Create tasks with concurrency limit
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(features):
            async with semaphore:
                return await send_prediction_request(session, url, features)
        
        tasks = [bounded_request(features) for features in requests_data]
        results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    # Process results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    latencies = [r['latency_ms'] for r in successful if r['latency_ms'] is not None]
    
    # Calculate stats
    stats = {
        'total_requests': num_requests,
        'successful': len(successful),
        'failed': len(failed),
        'total_time_sec': total_time,
        'throughput_req_per_sec': num_requests / total_time if total_time > 0 else 0,
    }
    
    if latencies:
        stats.update({
            'latency_min_ms': min(latencies),
            'latency_max_ms': max(latencies),
            'latency_mean_ms': np.mean(latencies),
            'latency_median_ms': np.median(latencies),
            'latency_p50_ms': np.percentile(latencies, 50),
            'latency_p95_ms': np.percentile(latencies, 95),
            'latency_p99_ms': np.percentile(latencies, 99),
            'latency_std_ms': np.std(latencies),
        })
    
    return stats, results


def print_results(stats: Dict, results: List[Dict]):
    """Print formatted results summary."""
    print(f"\n\nLoad Test Results")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Status: {'PASS' if stats['failed'] == 0 else 'DEGRADED'}")
    print()
    print(f"Requests Overview")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Successful: {stats['successful']} ({stats['successful']*100/stats['total_requests']:.1f}%)")
    print(f"  Failed: {stats['failed']} ({stats['failed']*100/stats['total_requests']:.1f}%)")
    print()
    print(f"Timing")
    print(f"  Total time: {stats['total_time_sec']:.2f} seconds")
    print(f"  Throughput: {stats['throughput_req_per_sec']:.1f} req/sec")
    print()
    
    if 'latency_mean_ms' in stats:
        print(f"Latency (milliseconds)")
        print(f"  Min: {stats['latency_min_ms']:.2f} ms")
        print(f"  Mean: {stats['latency_mean_ms']:.2f} ms")
        print(f"  Median (P50): {stats['latency_p50_ms']:.2f} ms")
        print(f"  P95: {stats['latency_p95_ms']:.2f} ms")
        print(f"  P99: {stats['latency_p99_ms']:.2f} ms")
        print(f"  Max: {stats['latency_max_ms']:.2f} ms")
        print(f"  Std Dev: {stats['latency_std_ms']:.2f} ms")
    
    # Error summary
    if stats['failed'] > 0:
        print()
        print(f"Error Summary")
        error_types = {}
        for r in results:
            if not r['success']:
                error = r.get('error', 'Unknown')
                error_types[error] = error_types.get(error, 0) + 1
        for error, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {error}: {count}")
    
    print("=" * 80)


def save_results_csv(stats: Dict, filepath: str = 'load_test_results.csv'):
    """Save results to CSV for historical tracking."""
    import csv
    
    row = {
        'timestamp': datetime.now().isoformat(),
        'total_requests': stats['total_requests'],
        'successful': stats['successful'],
        'failed': stats['failed'],
        'total_time_sec': stats['total_time_sec'],
        'throughput_req_per_sec': stats['throughput_req_per_sec'],
        'latency_mean_ms': stats.get('latency_mean_ms', None),
        'latency_p50_ms': stats.get('latency_p50_ms', None),
        'latency_p95_ms': stats.get('latency_p95_ms', None),
        'latency_p99_ms': stats.get('latency_p99_ms', None),
    }
    
    # Check if file exists to determine if we write header
    import os
    file_exists = os.path.isfile(filepath)
    
    try:
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writeheader()
            writer.writerow(row)
        print(f"\nResults saved to {filepath}")
    except Exception as e:
        print(f"Warning: Could not save results to CSV: {e}")


def main():
    parser = argparse.ArgumentParser(description='Load test FastAPI prediction service')
    parser.add_argument('--host', default='localhost', help='API host (default: localhost)')
    parser.add_argument('--port', type=int, default=8000, help='API port (default: 8000)')
    parser.add_argument('--requests', type=int, default=500, 
                       help='Number of requests to send (default: 500)')
    parser.add_argument('--concurrency', type=int, default=10,
                       help='Number of concurrent requests (default: 10)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("FOOTBALL PLAYER MARKET VALUE PREDICTOR - LOAD TEST")
    print("=" * 80)
    
    try:
        # Run load test
        stats, results = asyncio.run(
            run_load_test(args.host, args.port, args.requests, args.concurrency)
        )
        
        # Print results
        print_results(stats, results)
        
        # Save to CSV
        save_results_csv(stats)
        
    except Exception as e:
        print(f"\nError running load test: {e}")
        print(f"Is the API running? Try: uvicorn app:app --host {args.host} --port {args.port}")
        raise


if __name__ == "__main__":
    main()
