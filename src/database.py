import sqlite3
from pathlib import Path

def save_df_to_sqlite(df, db_path, table_name, if_exists="replace"):
    """
    Saves pandas dataframe into a SQLite database
    """
    
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn= sqlite3.connect(db_path)
    
    df.to_sql(
        table_name, 
        conn, 
        if_exists=if_exists,
        index=False
    )
    
    conn.close()
    
    print(f"Saved {len(df)} rows to {db_path} table: {table_name}")