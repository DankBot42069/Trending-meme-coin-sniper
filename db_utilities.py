#!/usr/bin/env python3
"""
Database Utilities Module

This module provides advanced utilities for working with the trading bot database,
including data analysis, reporting, maintenance, and optimization functions.

It serves as both a practical tool and an educational example of database interaction
patterns in Python, demonstrating best practices for SQLite database management.

Usage example:
    python db_utilities.py --analyze-trades --optimize

Author: Claude
Date: March 2025
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple, Set, TypedDict, cast
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from pathlib import Path

# Import the database module
import database as db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("db_utilities.log")
    ]
)
logger = logging.getLogger(__name__)


class TradeAnalysis:
    """
    Utility class for analyzing trade data from the database.
    
    This class provides methods to:
    1. Calculate trade performance metrics
    2. Identify profitable tokens and trading patterns
    3. Generate trading reports and visualizations
    4. Export data for external analysis
    
    It demonstrates advanced SQL queries and data processing techniques.
    """
    
    def __init__(self, conn: Optional[sqlite3.Connection] = None) -> None:
        """
        Initialize the TradeAnalysis with optional database connection.
        
        If no connection is provided, a new one will be created when needed.
        
        Args:
            conn: Optional database connection to use
        """
        self.conn = conn
    
    def _get_conn(self) -> sqlite3.Connection:
        """
        Get a database connection.
        
        Returns the existing connection if available, or creates a new one.
        
        Returns:
            SQLite database connection
        """
        if self.conn is None:
            self.conn = db.get_conn()
        return self.conn
    
    def get_profit_summary(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for all trades.
        
        Returns a dictionary with metrics like:
        - Total profit/loss in USD
        - Number of profitable vs. unprofitable trades
        - Average profit/loss per trade
        - Best and worst performing tokens
        
        Returns:
            Dictionary with profit summary statistics
        """
        conn = self._get_conn()
        
        # Calculate total profit/loss
        query_total = "SELECT SUM(CAST(pnl_usd AS REAL)) FROM trades"
        total_pnl = conn.execute(query_total).fetchone()[0] or 0
        
        # Count profitable vs. unprofitable trades
        query_profitable = "SELECT COUNT(*) FROM trades WHERE CAST(pnl_usd AS REAL) > 0"
        profitable_count = conn.execute(query_profitable).fetchone()[0] or 0
        
        query_unprofitable = "SELECT COUNT(*) FROM trades WHERE CAST(pnl_usd AS REAL) <= 0"
        unprofitable_count = conn.execute(query_unprofitable).fetchone()[0] or 0
        
        # Calculate average profit/loss per trade
        query_avg = "SELECT AVG(CAST(pnl_usd AS REAL)) FROM trades"
        avg_pnl = conn.execute(query_avg).fetchone()[0] or 0
        
        # Find best performing token
        query_best = """
            SELECT token_address, SUM(CAST(pnl_usd AS REAL)) as total_pnl
            FROM trades
            GROUP BY token_address
            ORDER BY total_pnl DESC
            LIMIT 1
        """
        best_result = conn.execute(query_best).fetchone()
        best_token = {
            "address": best_result[0] if best_result else None,
            "total_pnl": best_result[1] if best_result else 0
        }
        
        # Find worst performing token
        query_worst = """
            SELECT token_address, SUM(CAST(pnl_usd AS REAL)) as total_pnl
            FROM trades
            GROUP BY token_address
            ORDER BY total_pnl ASC
            LIMIT 1
        """
        worst_result = conn.execute(query_worst).fetchone()
        worst_token = {
            "address": worst_result[0] if worst_result else None,
            "total_pnl": worst_result[1] if worst_result else 0
        }
        
        return {
            "total_pnl_usd": total_pnl,
            "profitable_trades": profitable_count,
            "unprofitable_trades": unprofitable_count,
            "win_rate": profitable_count / (profitable_count + unprofitable_count) if (profitable_count + unprofitable_count) > 0 else 0,
            "avg_pnl_usd": avg_pnl,
            "best_token": best_token,
            "worst_token": worst_token
        }
    
    def get_daily_performance(self, days: int = 30) -> pd.DataFrame:
        """
        Calculate daily trading performance over a specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            DataFrame with daily performance metrics
        """
        conn = self._get_conn()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query daily performance
        query = """
            SELECT 
                DATE(trade_time) as date,
                COUNT(*) as trade_count,
                SUM(CAST(pnl_usd AS REAL)) as total_pnl_usd,
                AVG(CAST(pnl_usd AS REAL)) as avg_pnl_usd,
                SUM(CASE WHEN CAST(pnl_usd AS REAL) > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN CAST(pnl_usd AS REAL) <= 0 THEN 1 ELSE 0 END) as losing_trades
            FROM trades
            WHERE trade_time >= ?
            GROUP BY DATE(trade_time)
            ORDER BY DATE(trade_time)
        """
        
        # Execute query and convert to DataFrame
        df = pd.read_sql_query(query, conn, params=(start_date.isoformat(),))
        
        # Add win rate column
        df['win_rate'] = df['winning_trades'] / df['trade_count']
        
        # Add cumulative PnL column
        df['cumulative_pnl_usd'] = df['total_pnl_usd'].cumsum()
        
        return df
    
    def plot_performance(self, days: int = 30, save_path: Optional[str] = None) -> None:
        """
        Create plots of trading performance.
        
        Args:
            days: Number of days to analyze
            save_path: Optional path to save the plots
        """
        # Get daily performance data
        daily_data = self.get_daily_performance(days)
        
        if daily_data.empty:
            print("No trading data available for the specified period.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot 1: Daily PnL
        axes[0].bar(daily_data['date'], daily_data['total_pnl_usd'], color=['g' if x > 0 else 'r' for x in daily_data['total_pnl_usd']])
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].set_title('Daily Profit/Loss (USD)')
        axes[0].set_ylabel('Profit/Loss (USD)')
        
        # Plot 2: Cumulative PnL
        axes[1].plot(daily_data['date'], daily_data['cumulative_pnl_usd'], color='blue', marker='o')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_title('Cumulative Profit/Loss (USD)')
        axes[1].set_ylabel('Cumulative Profit/Loss (USD)')
        
        # Plot 3: Win Rate
        axes[2].bar(daily_data['date'], daily_data['win_rate'], color='purple')
        axes[2].axhline(y=0.5, color='black', linestyle='--', linewidth=0.5)
        axes[2].set_title('Daily Win Rate')
        axes[2].set_ylabel('Win Rate')
        axes[2].set_ylim(0, 1)
        
        # Format x-axis
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_trade_report(self, days: int = 30, format: str = 'text') -> Union[str, Dict[str, Any]]:
        """
        Generate a comprehensive trading report.
        
        Args:
            days: Number of days to include in the report
            format: Output format ('text' or 'json')
            
        Returns:
            Trading report in the specified format
        """
        # Get overall profit summary
        summary = self.get_profit_summary()
        
        # Get daily performance
        daily_data = self.get_daily_performance(days)
        
        # Get top performing tokens
        conn = self._get_conn()
        query_top_tokens = """
            SELECT 
                token_address,
                COUNT(*) as trade_count,
                SUM(CAST(pnl_usd AS REAL)) as total_pnl_usd,
                AVG(CAST(pnl_usd AS REAL)) as avg_pnl_usd
            FROM trades
            WHERE trade_time >= ?
            GROUP BY token_address
            ORDER BY total_pnl_usd DESC
            LIMIT 5
        """
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        top_tokens_df = pd.read_sql_query(query_top_tokens, conn, params=(start_date,))
        
        # Format report based on requested output format
        if format == 'json':
            return {
                "report_date": datetime.now().isoformat(),
                "period_days": days,
                "summary": summary,
                "daily_performance": daily_data.to_dict(orient='records'),
                "top_tokens": top_tokens_df.to_dict(orient='records')
            }
        else:  # text format
            # Build text report
            report = []
            report.append("=" * 80)
            report.append(f"TRADING PERFORMANCE REPORT - {datetime.now().strftime('%Y-%m-%d')}")
            report.append("=" * 80)
            
            report.append("\nOVERALL SUMMARY:")
            report.append(f"Total P&L: ${summary['total_pnl_usd']:.2f}")
            report.append(f"Win Rate: {summary['win_rate']:.2%} ({summary['profitable_trades']} winning trades, {summary['unprofitable_trades']} losing trades)")
            report.append(f"Average P&L per Trade: ${summary['avg_pnl_usd']:.2f}")
            
            report.append("\nBEST PERFORMING TOKEN:")
            report.append(f"Address: {summary['best_token']['address']}")
            report.append(f"Total P&L: ${summary['best_token']['total_pnl']:.2f}")
            
            report.append("\nWORST PERFORMING TOKEN:")
            report.append(f"Address: {summary['worst_token']['address']}")
            report.append(f"Total P&L: ${summary['worst_token']['total_pnl']:.2f}")
            
            report.append("\nTOP 5 TOKENS (LAST 30 DAYS):")
            for _, row in top_tokens_df.iterrows():
                report.append(f"- {row['token_address']}: ${row['total_pnl_usd']:.2f} ({row['trade_count']} trades)")
            
            report.append("\nDAILY PERFORMANCE (LAST 7 DAYS):")
            recent_days = daily_data.tail(7).copy()
            for _, row in recent_days.iterrows():
                report.append(f"- {row['date']}: ${row['total_pnl_usd']:.2f} ({row['trade_count']} trades, {row['win_rate']:.2%} win rate)")
            
            report.append("\n" + "=" * 80)
            
            return "\n".join(report)
    
    def export_trade_data(self, output_path: str, days: Optional[int] = None) -> bool:
        """
        Export trade data to CSV for external analysis.
        
        Args:
            output_path: Path to save the CSV file
            days: Optional number of days to limit the export
            
        Returns:
            True if export was successful, False otherwise
        """
        conn = self._get_conn()
        
        try:
            # Build query
            query = "SELECT * FROM trades"
            params = None
            
            if days is not None:
                start_date = (datetime.now() - timedelta(days=days)).isoformat()
                query += " WHERE trade_time >= ?"
                params = (start_date,)
            
            # Execute query and save to CSV
            df = pd.read_sql_query(query, conn, params=params)
            df.to_csv(output_path, index=False)
            
            print(f"Exported {len(df)} trades to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting trade data: {e}")
            return False


class DatabaseMaintenance:
    """
    Utility class for database maintenance operations.
    
    This class provides methods to:
    1. Optimize the database
    2. Check for and repair integrity issues
    3. Vacuum the database to recover space
    4. Manage indexes for performance
    5. Clean up old or redundant data
    
    It demonstrates best practices for SQLite database maintenance.
    """
    
    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize the DatabaseMaintenance with optional database path.
        
        Args:
            db_path: Optional path to the database file
        """
        self.db_path = db_path or str(db.DB_PATH)
    
    def optimize_database(self) -> bool:
        """
        Perform various optimization operations on the database.
        
        This includes:
        - Running ANALYZE to update statistics
        - Running VACUUM to reclaim space
        - Setting recommended PRAGMA settings
        
        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Run ANALYZE to update statistics
            conn.execute("ANALYZE")
            
            # Set recommended PRAGMA settings
            conn.execute("PRAGMA optimize")
            conn.execute("PRAGMA automatic_index = ON")
            conn.execute("PRAGMA cache_size = 10000")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            
            # Run VACUUM to reclaim space
            conn.execute("VACUUM")
            
            conn.close()
            
            logger.info("Database optimization completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False
    
    def check_integrity(self) -> List[str]:
        """
        Check the integrity of the database.
        
        Returns:
            List of integrity issues found, or empty list if none
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchall()
            
            # Check foreign key constraints
            cursor.execute("PRAGMA foreign_key_check")
            fk_result = cursor.fetchall()
            
            conn.close()
            
            issues = []
            
            # Process integrity check results
            if len(result) == 1 and result[0][0] == "ok":
                logger.info("Database integrity check passed")
            else:
                logger.warning("Database integrity check failed")
                for row in result:
                    issues.append(f"Integrity issue: {row[0]}")
            
            # Process foreign key check results
            if fk_result:
                logger.warning("Foreign key constraints violated")
                for row in fk_result:
                    issues.append(f"Foreign key violation in table {row[0]}, rowid {row[1]}")
            
            return issues
        except Exception as e:
            logger.error(f"Error checking database integrity: {e}")
            return [f"Error: {str(e)}"]
    
    def repair_database(self) -> bool:
        """
        Attempt to repair database integrity issues.
        
        Returns:
            True if repair was successful, False otherwise
        """
        try:
            # Check if database file exists
            if not os.path.exists(self.db_path):
                logger.error(f"Database file not found: {self.db_path}")
                return False
            
            # Create backup before repair
            backup_path = f"{self.db_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(self.db_path, 'rb') as src, open(backup_path, 'wb') as dst:
                dst.write(src.read())
            
            logger.info(f"Created backup at {backup_path}")
            
            # Check integrity first
            issues = self.check_integrity()
            if not issues:
                logger.info("No integrity issues found, no repair needed")
                return True
            
            # Simple repair approach: export and reimport data
            # This is a basic approach - more sophisticated repairs would depend on the specific issues
            
            # Connect to original database
            conn_orig = sqlite3.connect(self.db_path)
            
            # Create new database
            temp_db_path = f"{self.db_path}.new"
            
            # Initialize schema in new database
            db.DB_PATH = Path(temp_db_path)
            db.db_init()
            
            # Reset back original path for now
            db.DB_PATH = Path(self.db_path)
            
            # Copy data table by table
            tables = [
                "bought_tokens",
                "trades",
                "sold_tokens",
                "token_test_results",
                "seen_addresses",
                "low_value_tokens",
                "all_tokens",
                "tokens_full_details",
                "do_not_trade",
                "trailing_data"
            ]
            
            conn_new = sqlite3.connect(temp_db_path)
            
            for table in tables:
                try:
                    # Export data
                    query = f"SELECT * FROM {table}"
                    df = pd.read_sql_query(query, conn_orig)
                    
                    if not df.empty:
                        # Import to new database
                        df.to_sql(table, conn_new, if_exists='replace', index=False)
                        logger.info(f"Transferred {len(df)} records from table {table}")
                except Exception as table_error:
                    logger.warning(f"Error transferring table {table}: {table_error}")
            
            # Close connections
            conn_orig.close()
            conn_new.close()
            
            # Replace original with new database
            os.remove(self.db_path)
            os.rename(temp_db_path, self.db_path)
            
            # Set back the original path
            db.DB_PATH = Path(self.db_path)
            
            # Verify repair
            new_issues = self.check_integrity()
            if not new_issues:
                logger.info("Database repair successful")
                return True
            else:
                logger.error("Database repair failed, issues remain")
                logger.error(f"Issues: {new_issues}")
                return False
        except Exception as e:
            logger.error(f"Error repairing database: {e}")
            return False
    
    def clean_old_data(self, days: int = 90) -> int:
        """
        Clean up old data from the database to improve performance.
        
        Args:
            days: Age threshold in days for data to be considered old
            
        Returns:
            Number of records removed
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tables with timestamp columns to clean
            cleanup_targets = [
                {"table": "trades", "timestamp_col": "trade_time"},
                {"table": "sold_tokens", "timestamp_col": "timestamp"}
                # Add other tables as needed
            ]
            
            total_removed = 0
            
            for target in cleanup_targets:
                table = target["table"]
                timestamp_col = target["timestamp_col"]
                
                # Count records to be removed
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {timestamp_col} < ?", (cutoff_date,))
                count = cursor.fetchone()[0]
                
                if count > 0:
                    # Remove old records
                    cursor.execute(f"DELETE FROM {table} WHERE {timestamp_col} < ?", (cutoff_date,))
                    logger.info(f"Removed {count} old records from {table}")
                    total_removed += count
            
            # Commit changes
            conn.commit()
            
            # Optimize after removing data
            cursor.execute("VACUUM")
            
            conn.close()
            
            return total_removed
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")
            return 0
    
    def analyze_database_size(self) -> Dict[str, Any]:
        """
        Analyze database size and structure.
        
        Returns:
            Dictionary with size metrics
        """
        try:
            # Get file size
            file_size = os.path.getsize(self.db_path)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get size and row count for each table
            table_stats = {}
            total_rows = 0
            
            for table in tables:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                total_rows += row_count
                
                # Approximate table size (SQLite doesn't provide direct table size info)
                # This approach gets the average row size by sampling and multiplies by row count
                if row_count > 0:
                    # Get column count
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    column_count = len(columns)
                    
                    # Sample some rows to estimate size
                    sample_size = min(100, row_count)
                    cursor.execute(f"SELECT * FROM {table} LIMIT {sample_size}")
                    sample = cursor.fetchall()
                    
                    # Estimate average row size
                    total_sample_size = sum(len(str(row)) for row in sample)
                    avg_row_size = total_sample_size / sample_size
                    
                    # Estimate table size
                    estimated_size = avg_row_size * row_count
                else:
                    estimated_size = 0
                
                table_stats[table] = {
                    "row_count": row_count,
                    "estimated_size_bytes": estimated_size
                }
            
            conn.close()
            
            return {
                "file_size_bytes": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "total_rows": total_rows,
                "table_stats": table_stats
            }
        except Exception as e:
            logger.error(f"Error analyzing database size: {e}")
            return {"error": str(e)}


def main() -> None:
    """
    Main function to handle command-line arguments and execute utilities.
    """
    parser = argparse.ArgumentParser(description="Database utilities for trading bot")
    
    # Analysis flags
    parser.add_argument("--analyze-trades", action="store_true", help="Analyze trading performance")
    parser.add_argument("--report", action="store_true", help="Generate trading report")
    parser.add_argument("--plot", action="store_true", help="Create performance plots")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    parser.add_argument("--export", type=str, help="Export trade data to CSV")
    
    # Maintenance flags
    parser.add_argument("--optimize", action="store_true", help="Optimize database")
    parser.add_argument("--check", action="store_true", help="Check database integrity")
    parser.add_argument("--repair", action="store_true", help="Attempt to repair database issues")
    parser.add_argument("--clean", action="store_true", help="Clean old data")
    parser.add_argument("--analyze-size", action="store_true", help="Analyze database size")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize database if it doesn't exist
    import config as c
    # Ensure db is initialized with the correct path
    db.DB_PATH = Path(c.DB_PATH)
    db.db_init()
    
    # Handle analysis flags
    if args.analyze_trades or args.report or args.plot or args.export:
        trade_analysis = TradeAnalysis()
        
        if args.analyze_trades:
            summary = trade_analysis.get_profit_summary()
            print("\n=== TRADING PERFORMANCE SUMMARY ===")
            print(f"Total P&L: ${summary['total_pnl_usd']:.2f}")
            print(f"Win Rate: {summary['win_rate']:.2%} ({summary['profitable_trades']} winning trades, {summary['unprofitable_trades']} losing trades)")
            print(f"Average P&L per Trade: ${summary['avg_pnl_usd']:.2f}")
        
        if args.report:
            report = trade_analysis.generate_trade_report(days=args.days)
            print(report)
        
        if args.plot:
            trade_analysis.plot_performance(days=args.days)
        
        if args.export:
            trade_analysis.export_trade_data(args.export, days=args.days)
    
    # Handle maintenance flags
    if args.optimize or args.check or args.repair or args.clean or args.analyze_size:
        maintenance = DatabaseMaintenance()
        
        if args.optimize:
            print("Optimizing database...")
            success = maintenance.optimize_database()
            print("Optimization completed successfully" if success else "Optimization failed")
        
        if args.check:
            print("Checking database integrity...")
            issues = maintenance.check_integrity()
            if issues:
                print(f"Found {len(issues)} integrity issues:")
                for issue in issues:
                    print(f"- {issue}")
            else:
                print("No integrity issues found.")
        
        if args.repair:
            print("Attempting to repair database...")
            success = maintenance.repair_database()
            print("Repair completed successfully" if success else "Repair failed")
        
        if args.clean:
            print(f"Cleaning old data (older than {args.days} days)...")
            removed = maintenance.clean_old_data(days=args.days)
            print(f"Removed {removed} old records")
        
        if args.analyze_size:
            print("Analyzing database size...")
            size_info = maintenance.analyze_database_size()
            
            print(f"Database file size: {size_info['file_size_mb']:.2f} MB")
            print(f"Total rows: {size_info['total_rows']}")
            
            print("\nTable statistics:")
            for table, stats in size_info['table_stats'].items():
                print(f"- {table}: {stats['row_count']} rows, ~{stats['estimated_size_bytes'] / 1024:.2f} KB")
    
    # If no arguments were provided, print help
    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    main()