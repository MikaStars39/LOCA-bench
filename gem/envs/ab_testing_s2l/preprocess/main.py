from argparse import ArgumentParser
import os
import sys
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from gem.utils.filesystem import nfs_safe_rmtree

# Add current directory to path for local module imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Add mcp_convert path for GoogleCloudDatabase
from mcp_convert.mcps.google_cloud.database_utils import GoogleCloudDatabase


def clean_dataset(db: GoogleCloudDatabase, project_id: str) -> bool:
    """Clean and setup BigQuery dataset for ab_testing"""
    print("=" * 60)
    print("BigQuery Dataset Management for A/B Testing Task")
    print("=" * 60)

    dataset_id = "ab_testing"

    try:
        # Check if dataset exists
        print(f"\n1. Checking if dataset '{dataset_id}' exists...")
        existing_dataset = db.get_bigquery_dataset(project_id, dataset_id)

        if existing_dataset:
            print(f"   ✅ Dataset '{dataset_id}' exists - deleting...")
            # Delete all tables in the dataset first
            tables = db.list_bigquery_tables(project_id, dataset_id)
            for table in tables:
                table_id = table['tableId']
                db.delete_bigquery_table(project_id, dataset_id, table_id)
                print(f"      ✓ Deleted table: {table_id}")

            # Delete the dataset
            db.delete_bigquery_dataset(project_id, dataset_id)
            print(f"   ✅ Dataset '{dataset_id}' deleted")
        else:
            print(f"   ℹ️  Dataset '{dataset_id}' does not exist")

        # Create new dataset
        print(f"\n2. Creating new dataset '{dataset_id}'...")
        dataset_info = {
            "location": "US",
            "description": "A/B testing dataset for conversion rate analysis",
            "labels": {}
        }

        success = db.create_bigquery_dataset(project_id, dataset_id, dataset_info)

        if success:
            print(f"   ✅ Dataset '{dataset_id}' created successfully in US")
        else:
            print(f"   ❌ Failed to create dataset '{dataset_id}'")
            return False

        # List all datasets to verify
        print(f"\n3. Listing all datasets...")
        datasets = db.list_bigquery_datasets()
        if datasets:
            print(f"   Datasets in project '{project_id}':")
            for ds in datasets:
                print(f"      - {ds['datasetId']}")
        else:
            print("   No datasets found")

        print("\n✅ Dataset management complete!")
        print(f"   Ready to populate dataset '{dataset_id}' with A/B testing CSV files.")
        return True

    except Exception as e:
        print(f"❌ Error in dataset cleanup: {e}")
        import traceback
        traceback.print_exc()
        return False


def clean_bucket(db: GoogleCloudDatabase) -> bool:
    """Clean Cloud Storage bucket for ab_testing"""
    print("=" * 60)
    print("Cloud Storage Management for A/B Testing Task")
    print("=" * 60)

    bucket_name = "promo-assets-for-b"

    try:
        # Check if bucket exists
        print(f"\n1. Checking if bucket '{bucket_name}' exists...")
        existing_bucket = db.get_storage_bucket(bucket_name)

        if existing_bucket:
            print(f"   📦 Found existing bucket: {bucket_name}")
            print(f"   🗑️  Deleting bucket: {bucket_name} and all its contents...")

            # Delete all objects in the bucket first
            objects = db.list_storage_objects(bucket_name)
            for obj in objects:
                db.delete_storage_object(bucket_name, obj['name'])
                print(f"      ✓ Deleted object: {obj['name']}")

            # Delete the bucket
            db.delete_storage_bucket(bucket_name)
            print(f"   ✅ Successfully deleted bucket: {bucket_name}")
        else:
            print(f"   ✅ Bucket {bucket_name} does not exist - no cleanup needed")

        # List all storage buckets to verify
        print(f"\n2. Listing all storage buckets...")
        buckets = db.list_storage_buckets()
        if buckets:
            print(f"   Storage buckets:")
            for bucket in buckets:
                print(f"      - {bucket['name']} (location: {bucket.get('location', 'Unknown')})")
        else:
            print("   No storage buckets found")

        print("\n✅ Storage bucket management complete!")
        return True

    except Exception as e:
        print(f"❌ Error in bucket cleanup: {e}")
        import traceback
        traceback.print_exc()
        return False


def clean_log(db: GoogleCloudDatabase) -> bool:
    """Clean and setup Cloud Logging for ab_testing"""
    print("=" * 60)
    print("Cloud Logging Management for A/B Testing Task")
    print("=" * 60)

    print("\n✅ Log bucket management complete (simulated in local DB)")
    print("   Ready to write logs to 'abtesting_logging'")
    return True


def generate_ab_test_data(task_root: Path,
                          num_scenarios: int = 20,
                          num_days: int = 15,
                          difficulty: str = "medium",
                          seed: int = 42,
                          **kwargs) -> Dict:
    """Generate A/B test data using the numpy-vectorized generator.

    Args:
        task_root: Task root directory (task_dir, where data will be generated)
        num_scenarios: Number of scenarios to generate
        num_days: Number of days per scenario
        difficulty: Difficulty level (easy/medium/hard)
        seed: Random seed
        **kwargs: Additional parameters for the generator

    Returns:
        Result dict from generate_scenarios (contains 'scenarios' with numpy arrays),
        or None on failure.
    """
    print("=" * 60)
    print("Generating A/B Test Data (numpy-vectorized)")
    print("=" * 60)

    try:
        # Import generator directly
        code_dir = Path(__file__).parent.parent
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from generate_ab_data import ABTestingDataGenerator

        print(f"🎲 Generation parameters:")
        print(f"   Scenarios: {num_scenarios}")
        print(f"   Days per scenario: {num_days}")
        print(f"   Rows per scenario: {num_days * 24}")
        print(f"   Difficulty: {difficulty}")
        print(f"   Seed: {seed}")

        generator = ABTestingDataGenerator(seed=seed)

        # Map kwargs to generate_scenarios parameters
        gen_kwargs = {}
        base_conv_min = kwargs.get('base_conversion_min', 0.70)
        base_conv_max = kwargs.get('base_conversion_max', 0.78)
        gen_kwargs['base_conversion_range'] = (base_conv_min, base_conv_max)

        conv_diff_min = kwargs.get('conversion_diff_min', -0.03)
        conv_diff_max = kwargs.get('conversion_diff_max', 0.03)
        gen_kwargs['conversion_diff_range'] = (conv_diff_min, conv_diff_max)

        click_min = kwargs.get('click_min', 0)
        click_max = kwargs.get('click_max', 200)
        gen_kwargs['click_range'] = (click_min, click_max)

        gen_kwargs['noise_level'] = kwargs.get('noise_level', 0.1)
        gen_kwargs['zero_probability'] = kwargs.get('zero_probability', 0.05)

        result = generator.generate_scenarios(
            num_scenarios=num_scenarios,
            num_days=num_days,
            difficulty=difficulty,
            **gen_kwargs
        )

        # Create output directory and clean old CSV files
        output_dir = task_root / "files"
        if output_dir.exists():
            old_csv_files = list(output_dir.glob("ab_*.csv"))
            for old_file in old_csv_files:
                old_file.unlink()
        output_dir.mkdir(exist_ok=True, parents=True)

        # Write CSV files in parallel using ThreadPoolExecutor
        scenarios = result["scenarios"]

        def _write_csv(scenario):
            filename = f"ab_{scenario['name']}.csv"
            generator.save_csv(scenario["arrays"], output_dir / filename)

        max_workers = min(8, len(scenarios))
        if len(scenarios) > 50:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(_write_csv, scenarios))
        else:
            for s in scenarios:
                _write_csv(s)

        # Save ground truth
        groundtruth_dir = task_root / "groundtruth_workspace"
        groundtruth_dir.mkdir(exist_ok=True, parents=True)
        expected_ratio_file = groundtruth_dir / "expected_ratio.csv"
        generator.save_expected_ratio(scenarios, expected_ratio_file)

        print(f"✅ Generated {result['num_scenarios']} scenarios, "
              f"{result['num_scenarios'] * result['num_days'] * 24} total rows")
        print("✅ Data generation successful!")
        return result

    except Exception as e:
        print(f"❌ Data generation error: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_scenarios_to_bigquery(scenarios: List[Dict],
                                db: GoogleCloudDatabase,
                                project_id: str,
                                dataset_id: str) -> bool:
    """Insert scenario data directly from numpy arrays into SQLite.

    Bypasses CSV round-trip entirely — goes from in-memory numpy arrays
    straight to SQLite via executemany.

    Args:
        scenarios: List of scenario dicts with 'arrays' key (numpy arrays)
        db: GoogleCloudDatabase instance
        project_id: BigQuery project ID
        dataset_id: BigQuery dataset ID

    Returns:
        True on success
    """
    print("=" * 60)
    print("Loading scenarios directly to BigQuery (memory → SQLite)")
    print("=" * 60)

    try:
        conn = db.sqlite._get_connection()
        # Tune SQLite for bulk loading
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA cache_size=-64000")  # 64 MB

        schema = [
            {"name": "time_window", "type": "STRING", "mode": "NULLABLE"},
            {"name": "A_clicks", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "A_store_views", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "B_clicks", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "B_store_views", "type": "INTEGER", "mode": "NULLABLE"},
        ]

        all_tables = db.json_db.load_data(db.bigquery_tables_file) or {}

        for idx, scenario in enumerate(scenarios):
            table_name = f"ab_{scenario['name']}".replace("-", "_").replace(" ", "_")
            arrays = scenario["arrays"]

            # Create table (no per-table commit)
            db.sqlite.create_table_from_schema(
                project_id, dataset_id, table_name, schema, auto_commit=False)

            # Build tuples directly from numpy arrays
            rows_tuples = list(zip(
                arrays["time_windows"],
                arrays["A_clicks"].tolist(),
                arrays["A_store_views"].tolist(),
                arrays["B_clicks"].tolist(),
                arrays["B_store_views"].tolist(),
            ))

            # Direct executemany — skip dict-cleaning overhead in insert_rows()
            full_table = db.sqlite._get_table_name(project_id, dataset_id, table_name)
            conn.executemany(
                f'INSERT INTO {full_table} '
                f'("time_window","A_clicks","A_store_views","B_clicks","B_store_views") '
                f'VALUES (?,?,?,?,?)',
                rows_tuples
            )

            # Build metadata
            now_str = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            key = f"{project_id}:{dataset_id}.{table_name}"
            all_tables[key] = {
                "schema": schema,
                "tableId": table_name,
                "datasetId": dataset_id,
                "projectId": project_id,
                "created": now_str,
                "modified": now_str,
                "numRows": len(rows_tuples),
                "numBytes": 0,
                "description": f"A/B testing table from ab_{scenario['name']}.csv",
            }

            if (idx + 1) % 500 == 0:
                print(f"   ... loaded {idx + 1}/{len(scenarios)} tables")

        conn.commit()
        conn.execute("PRAGMA synchronous=FULL")

        # Single JSON write for all table metadata
        db.json_db.save_data(db.bigquery_tables_file, all_tables)
        db.json_db.save_data(db.query_results_file, {})

        print(f"✅ Loaded {len(scenarios)} tables directly into SQLite")
        return True

    except Exception as e:
        print(f"❌ Error loading scenarios to BigQuery: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_csvs_to_bigquery(db: GoogleCloudDatabase,
                            project_id: str,
                            dataset_id: str,
                            csv_folder: str,
                            csv_pattern: str = "*.csv") -> bool:
    """Upload CSV files to BigQuery tables in local database (batch mode).

    Legacy function kept for backward compatibility. Prefer
    load_scenarios_to_bigquery() for new code.
    """
    print("=" * 60)
    print("Uploading CSV Files to BigQuery")
    print("=" * 60)

    try:
        # Find all CSV files
        import glob
        csv_files = glob.glob(os.path.join(csv_folder, csv_pattern))

        if not csv_files:
            print(f"❌ No CSV files found matching pattern {csv_pattern} in {csv_folder}")
            return False

        print(f"\n📁 Found {len(csv_files)} CSV files to upload")

        # Load JSON metadata once before the loop
        all_tables = db.json_db.load_data(db.bigquery_tables_file)
        if not isinstance(all_tables, dict):
            all_tables = {}

        # Upload each CSV file (using direct SQLite calls without per-table commits)
        for csv_file in csv_files:
            # Extract table name from filename (without extension)
            table_name = Path(csv_file).stem

            # Clean table name (BigQuery table names have restrictions)
            table_name = table_name.replace("-", "_").replace(" ", "_")

            print(f"\n📤 Uploading {Path(csv_file).name} -> {dataset_id}.{table_name}")

            try:
                # Read CSV file
                rows = []
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        # Convert numeric fields
                        converted_row = {}
                        for key, value in row.items():
                            # Try to convert to number
                            try:
                                if '.' in value:
                                    converted_row[key] = float(value)
                                else:
                                    converted_row[key] = int(value)
                            except (ValueError, AttributeError):
                                converted_row[key] = value
                        rows.append(converted_row)

                if not rows:
                    print(f"   ⚠️  No data in {csv_file}")
                    continue

                # Create schema from first row
                schema = []
                for key, value in rows[0].items():
                    if isinstance(value, int):
                        field_type = "INTEGER"
                    elif isinstance(value, float):
                        field_type = "FLOAT"
                    else:
                        field_type = "STRING"

                    schema.append({
                        "name": key,
                        "type": field_type,
                        "mode": "NULLABLE"
                    })

                # Create SQLite table directly (no commit per table)
                db.sqlite.create_table_from_schema(project_id, dataset_id, table_name,
                                                   schema, auto_commit=False)
                print(f"   ✓ Created table with {len(schema)} columns")

                # Insert rows directly into SQLite (no commit per table)
                inserted = db.sqlite.insert_rows(project_id, dataset_id, table_name,
                                                 rows, schema, auto_commit=False)

                if inserted > 0:
                    print(f"   ✅ Loaded {inserted} rows into {dataset_id}.{table_name}")
                else:
                    print(f"   ❌ Failed to insert rows into {table_name}")

                # Build metadata in memory dict
                key = f"{project_id}:{dataset_id}.{table_name}"
                now_str = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                table_meta = {
                    "schema": schema,
                    "description": f"A/B testing table from {Path(csv_file).name}",
                    "tableId": table_name,
                    "datasetId": dataset_id,
                    "projectId": project_id,
                    "created": now_str,
                    "numRows": 0,
                    "numBytes": 0,
                }
                if inserted > 0:
                    table_meta["numRows"] = inserted
                    table_meta["modified"] = now_str
                all_tables[key] = table_meta

            except Exception as e:
                print(f"   ❌ Error uploading {csv_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Single commit for all SQLite operations
        db.sqlite._get_connection().commit()
        # Single JSON write for all table metadata
        db.json_db.save_data(db.bigquery_tables_file, all_tables)
        # Clear query cache once
        db.json_db.save_data(db.query_results_file, {})

        print("\n✅ CSV upload complete!")
        return True

    except Exception as e:
        print(f"❌ Error in CSV upload: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--agent_workspace", required=False)
    parser.add_argument("--launch_time", required=False, help="Launch time")

    # Data generation parameters
    parser.add_argument("--skip-generation", action="store_true",
                       help="Skip data generation, use existing CSV files")
    parser.add_argument("--num-scenarios", type=int, default=50,
                       help="Number of scenarios to generate (default: 20)")
    parser.add_argument("--num-days", type=int, default=15,
                       help="Number of days per scenario (default: 15)")
    parser.add_argument("--difficulty", type=str, default="medium",
                       choices=["easy", "medium", "hard"],
                       help="Difficulty level (default: medium)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for data generation (default: 42)")

    # Advanced generation parameters
    parser.add_argument("--base-conversion-min", type=float, default=None)
    parser.add_argument("--base-conversion-max", type=float, default=None)
    parser.add_argument("--conversion-diff-min", type=float, default=None)
    parser.add_argument("--conversion-diff-max", type=float, default=None)
    parser.add_argument("--click-min", type=int, default=None)
    parser.add_argument("--click-max", type=int, default=None)
    parser.add_argument("--noise-level", type=float, default=None)
    parser.add_argument("--zero-probability", type=float, default=None)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🚀 A/B Testing Task Environment Preprocessing")
    print("=" * 60)
    print("Using local Google Cloud database")

    # Determine Google Cloud database directory
    if args.agent_workspace:
        workspace_parent = Path(args.agent_workspace).parent
        gcloud_db_dir = str(workspace_parent / "local_db" / "google_cloud")
    else:
        gcloud_db_dir = str(Path(__file__).parent.parent / "local_db" / "google_cloud")

    print(f"\n📂 Google Cloud Database Directory: {gcloud_db_dir}")

    # Clean up existing database directory before starting
    if Path(gcloud_db_dir).exists():
        print(f"🗑️  Cleaning existing database directory...")
        try:
            nfs_safe_rmtree(gcloud_db_dir)
            print(f"   ✓ Removed old database files")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not fully clean directory: {e}")

    # Create fresh database directory
    Path(gcloud_db_dir).mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Created fresh database directory")

    # Initialize GoogleCloudDatabase
    print("\n📊 Initializing Google Cloud Database...")
    gcloud_db = GoogleCloudDatabase(data_dir=gcloud_db_dir)

    # Use a default project ID for local database
    project_id = "local-project"
    print(f"   Using project: {project_id}")

    # Get task root directory from agent_workspace
    if args.agent_workspace:
        task_root = Path(args.agent_workspace).parent
    else:
        task_root = Path(__file__).parent.parent

    print(f"   Task root directory: {task_root}")

    # Step 0: Generate A/B test data
    result = None
    if not args.skip_generation:
        print("\n" + "=" * 60)
        print("STEP 0: Generate A/B Test Data")
        print("=" * 60)

        advanced_params = {}
        if args.base_conversion_min is not None:
            advanced_params['base_conversion_min'] = args.base_conversion_min
        if args.base_conversion_max is not None:
            advanced_params['base_conversion_max'] = args.base_conversion_max
        if args.conversion_diff_min is not None:
            advanced_params['conversion_diff_min'] = args.conversion_diff_min
        if args.conversion_diff_max is not None:
            advanced_params['conversion_diff_max'] = args.conversion_diff_max
        if args.click_min is not None:
            advanced_params['click_min'] = args.click_min
        if args.click_max is not None:
            advanced_params['click_max'] = args.click_max
        if args.noise_level is not None:
            advanced_params['noise_level'] = args.noise_level
        if args.zero_probability is not None:
            advanced_params['zero_probability'] = args.zero_probability

        result = generate_ab_test_data(
            task_root=task_root,
            num_scenarios=args.num_scenarios,
            num_days=args.num_days,
            difficulty=args.difficulty,
            seed=args.seed,
            **advanced_params
        )
        if result is None:
            print("❌ Data generation failed!")
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("STEP 0: Skip Data Generation")
        print("=" * 60)
        print("Using existing CSV files in files/ directory")

    # Step 1: Clean logs
    print("\n" + "=" * 60)
    print("STEP 1: Clean Log Buckets")
    print("=" * 60)
    clean_log(gcloud_db)

    # Step 2: Clean dataset
    print("\n" + "=" * 60)
    print("STEP 2: Clean BigQuery Dataset")
    print("=" * 60)
    if not clean_dataset(gcloud_db, project_id):
        print("❌ Dataset cleanup failed!")
        sys.exit(1)

    # Step 3: Clean bucket
    print("\n" + "=" * 60)
    print("STEP 3: Clean Cloud Storage Bucket")
    print("=" * 60)
    if not clean_bucket(gcloud_db):
        print("❌ Bucket cleanup failed!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("⏳ Configuration complete (no wait needed for local DB)")
    print("=" * 60)

    # Step 4: Load data into BigQuery
    print("\n" + "=" * 60)
    print("STEP 4: Load Data into BigQuery")
    print("=" * 60)

    if result is not None:
        # Direct memory → SQLite (fast path)
        if not load_scenarios_to_bigquery(
            result["scenarios"], gcloud_db, project_id, "ab_testing"
        ):
            print("❌ Direct loading failed!")
            sys.exit(1)
    else:
        # Fallback: load from CSV files (when --skip-generation was used)
        csv_folder = task_root / "files"
        if not csv_folder.exists():
            print(f"❌ CSV folder not found: {csv_folder}")
            sys.exit(1)
        if not upload_csvs_to_bigquery(
            db=gcloud_db,
            project_id=project_id,
            dataset_id="ab_testing",
            csv_folder=str(csv_folder),
            csv_pattern="*.csv"
        ):
            print("❌ CSV upload failed!")
            sys.exit(1)

    # Set environment variable for evaluation
    os.environ['GOOGLE_CLOUD_DATA_DIR'] = gcloud_db_dir

    # Write environment variable file
    env_file = Path(gcloud_db_dir).parent / ".gcloud_env"
    try:
        with open(env_file, 'w') as f:
            f.write(f"# Google Cloud Database Environment Variables\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"export GOOGLE_CLOUD_DATA_DIR={gcloud_db_dir}\n")
        print(f"\n📄 Environment variable file created: {env_file}")
    except Exception as e:
        print(f"⚠️  Unable to create environment variable file: {e}")

    print("\n" + "=" * 60)
    print("🎉 A/B Testing Task Environment Preprocessing Complete!")
    print("=" * 60)
    print(f"✅ Google Cloud database initialized")
    print(f"✅ BigQuery dataset 'ab_testing' created and populated")
    print(f"✅ Cloud Storage bucket cleaned")
    print(f"✅ Cloud Logging configured")

    # Count tables
    tables = gcloud_db.list_bigquery_tables(project_id, "ab_testing")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Tables created: {len(tables)}")
    for table in tables:
        print(f"      - {table['tableId']}: {table.get('numRows', 0)} rows")

    print(f"\n📂 Directory Locations:")
    print(f"   Google Cloud DB: {gcloud_db_dir}")
    if args.agent_workspace:
        print(f"   Agent Workspace: {args.agent_workspace}")

    print(f"\n📌 Environment Variables:")
    print(f"   GOOGLE_CLOUD_DATA_DIR={gcloud_db_dir}")

    print(f"\n💡 Next Step: Agent can now use google-cloud-simplified MCP server")
    print(f"   to query and analyze the A/B testing data")
