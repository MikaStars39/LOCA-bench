#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B Testing data generator — numpy-vectorized for speed.

Generates deterministic A/B test datasets using numpy arrays instead of
Python-loop-per-row dicts.  At 256k scale (5000 scenarios × 500 days =
60 M rows) this cuts generation time from ~300 s to ~3 s.
"""

import csv
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

import numpy as np


class ABTestingDataGenerator:
    """A/B Testing data generator (numpy-vectorized)."""

    # Scenario name pool (supports 100+ scenarios)
    SCENARIO_NAMES = [
        # E-commerce categories (20)
        "Appliances", "Automotive", "Baby", "Beauty", "Books",
        "Clothing", "Education", "Electronics", "Food", "FreshFood",
        "Gaming", "Health", "Home", "Hospitality", "Music",
        "Office", "Outdoor", "Pets", "Sports", "Travel",

        # More product categories (30)
        "Toys", "Garden", "Jewelry", "Furniture", "Art",
        "Crafts", "Industrial", "Software", "Movies", "VideoGames",
        "Fashion", "Shoes", "Bags", "Watches", "Eyewear",
        "Cosmetics", "Skincare", "Fragrance", "HairCare", "PersonalCare",
        "Kitchenware", "Bedding", "Bath", "Decor", "Lighting",
        "Tools", "Hardware", "Paint", "Plumbing", "Electrical",

        # Specialty categories (30)
        "Photography", "Audio", "Cameras", "Drones", "SmartHome",
        "Wearables", "Tablets", "Laptops", "Desktops", "Monitors",
        "Networking", "Storage", "Printers", "Scanners", "Projectors",
        "Musical", "Instruments", "RecordingEquipment", "DJEquipment", "ProAudio",
        "Fitness", "Yoga", "Cycling", "Running", "Swimming",
        "Camping", "Hiking", "Fishing", "Hunting", "Climbing",

        # Lifestyle categories (30)
        "Nutrition", "Supplements", "Vitamins", "Protein", "OrganicFood",
        "BabyFood", "BabyClothing", "BabyToys", "Diapers", "BabyCare",
        "PetFood", "PetToys", "PetCare", "PetGrooming", "PetTraining",
        "Wedding", "Party", "Gifts", "Flowers", "Cards",
        "Stationery", "SchoolSupplies", "OfficeSupplies", "ArtSupplies", "CraftSupplies",
        "Magazines", "Comics", "Audiobooks", "eBooks", "Textbooks",

        # Extended categories (25)
        "Antiques", "Collectibles", "Memorabilia", "VintageFashion", "VintageJewelry",
        "LuxuryGoods", "DesignerFashion", "HighEndElectronics", "PremiumBeauty", "GourmetFood",
        "Organic", "EcoFriendly", "Sustainable", "FairTrade", "LocalProducts",
        "HandmadeItems", "CustomProducts", "PersonalizedGifts", "BespokeServices", "ArtisanGoods",
        "DigitalProducts", "OnlineCourses", "Subscriptions", "Memberships", "VirtualGoods",

        # Extra product categories (30)
        "Snacks", "Beverages", "Coffee", "Tea", "Wine",
        "Beer", "Spirits", "Cheese", "Chocolate", "Bakery",
        "Seafood", "Meat", "Produce", "Dairy", "Frozen",
        "Canned", "Condiments", "Spices", "Pasta", "Rice",
        "Cereal", "Candy", "Desserts", "IceCream", "Pizza",
        "Sandwiches", "Salads", "Soups", "Sauces", "Dips",

        # Services and entertainment (30)
        "Streaming", "CloudServices", "WebHosting", "Security", "Insurance",
        "Banking", "Investment", "RealEstate", "Consulting", "Marketing",
        "Advertising", "Design", "Development", "Writing", "Translation",
        "Photography", "Videography", "Animation", "VoiceOver", "Podcast",
        "Events", "Catering", "Cleaning", "Maintenance", "Repair",
        "Installation", "Delivery", "Shipping", "Storage", "Moving",

        # Professional services (30)
        "Legal", "Accounting", "Tax", "Audit", "Compliance",
        "HR", "Recruitment", "Training", "Coaching", "Mentoring",
        "Therapy", "Counseling", "Nutrition", "Dietitian", "Fitness",
        "PersonalTraining", "Massage", "Spa", "Salon", "Barbershop",
        "Veterinary", "Grooming", "DayCare", "Tutoring", "MusicLessons",
        "DanceLessons", "ArtClasses", "LanguageLessons", "Workshops", "Seminars",

        # Hobbies (20)
        "Knitting", "Sewing", "Quilting", "Embroidery", "Crochet",
        "Woodworking", "Metalworking", "Pottery", "Painting", "Drawing",
        "Sculpting", "Photography", "Birdwatching", "Astronomy", "Gardening",
        "Aquariums", "Terrariums", "ModelBuilding", "Origami", "Calligraphy"
    ]

    def __init__(self, seed: int = 42):
        """Initialize generator with master seed."""
        random.seed(seed)
        self._np_seed = seed

    def generate_time_windows(self,
                              num_days: int = 15,
                              start_date: str = "7/29") -> List[str]:
        """Generate time window labels.

        Returns:
            List of strings like ["7/29 00:00-00:59", ...]
        """
        time_windows = []
        month, day = map(int, start_date.split('/'))
        current_date = datetime(2024, month, day)

        for _ in range(num_days):
            date_str = f"{current_date.month}/{current_date.day}"
            for hour in range(24):
                time_windows.append(f"{date_str} {hour:02d}:00-{hour:02d}:59")
            current_date += timedelta(days=1)

        return time_windows

    # ------------------------------------------------------------------
    # Vectorized data generation
    # ------------------------------------------------------------------

    def generate_ab_data(self,
                         time_windows: List[str],
                         base_conversion_rate: float = 0.74,
                         conversion_diff: float = 0.01,
                         click_range: Tuple[int, int] = (0, 200),
                         noise_level: float = 0.1,
                         zero_probability: float = 0.05,
                         rng: np.random.RandomState = None) -> Dict:
        """Generate A/B data as numpy arrays (vectorized).

        Returns dict with keys:
            time_windows  – list[str]
            A_clicks      – np.ndarray (int64)
            A_store_views – np.ndarray (int64)
            B_clicks      – np.ndarray (int64)
            B_store_views – np.ndarray (int64)
        """
        if rng is None:
            rng = np.random.RandomState()

        n = len(time_windows)
        a_conv = base_conversion_rate - conversion_diff / 2
        b_conv = base_conversion_rate + conversion_diff / 2

        # All random draws in bulk (no Python loop)
        zero_a = rng.random(n) < zero_probability
        zero_b = rng.random(n) < zero_probability
        a_clicks = rng.randint(click_range[0], click_range[1] + 1, size=n)
        b_clicks = rng.randint(click_range[0], click_range[1] + 1, size=n)
        a_noise = rng.normal(0, noise_level * a_conv, n)
        b_noise = rng.normal(0, noise_level * b_conv, n)

        a_actual = np.clip(a_conv + a_noise, 0.3, 0.95)
        b_actual = np.clip(b_conv + b_noise, 0.3, 0.95)
        a_views = (a_clicks * a_actual).astype(np.int64)
        b_views = (b_clicks * b_actual).astype(np.int64)

        a_clicks[zero_a] = 0; a_views[zero_a] = 0
        b_clicks[zero_b] = 0; b_views[zero_b] = 0

        return {
            "time_windows": time_windows,
            "A_clicks": a_clicks,
            "A_store_views": a_views,
            "B_clicks": b_clicks,
            "B_store_views": b_views,
        }

    def calculate_conversion_rate(self, arrays: Dict) -> Tuple[float, float]:
        """Calculate actual conversion rates from numpy arrays.

        Args:
            arrays: dict returned by generate_ab_data()

        Returns:
            (A_rate, B_rate)
        """
        total_a_clicks = int(arrays["A_clicks"].sum())
        total_a_views = int(arrays["A_store_views"].sum())
        total_b_clicks = int(arrays["B_clicks"].sum())
        total_b_views = int(arrays["B_store_views"].sum())

        a_rate = total_a_views / total_a_clicks if total_a_clicks > 0 else 0
        b_rate = total_b_views / total_b_clicks if total_b_clicks > 0 else 0

        return a_rate, b_rate

    def save_csv(self, arrays: Dict, output_file: Path):
        """Write scenario data from numpy arrays to CSV."""
        tw = arrays["time_windows"]
        ac = arrays["A_clicks"]
        av = arrays["A_store_views"]
        bc = arrays["B_clicks"]
        bv = arrays["B_store_views"]

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            f.write("time_window,A_clicks,A_store_views,B_clicks,B_store_views\n")
            for i in range(len(tw)):
                f.write(f"{tw[i]},{ac[i]},{av[i]},{bc[i]},{bv[i]}\n")

    # ------------------------------------------------------------------
    # Scenario generation (parallel-capable)
    # ------------------------------------------------------------------

    def generate_scenarios(self,
                           num_scenarios: int = 20,
                           num_days: int = 15,
                           base_conversion_range: Tuple[float, float] = (0.70, 0.78),
                           conversion_diff_range: Tuple[float, float] = (-0.03, 0.03),
                           click_range: Tuple[int, int] = (0, 200),
                           noise_level: float = 0.1,
                           zero_probability: float = 0.05,
                           difficulty: str = "medium") -> Dict:
        """Generate multiple scenarios with numpy arrays.

        Returns dict with keys: scenarios, num_scenarios, num_days,
        difficulty, parameters.  Each scenario contains 'arrays' (numpy)
        instead of 'data_rows' (list-of-dicts).
        """
        # Adjust params by difficulty
        if difficulty == "easy":
            conversion_diff_range = (0.02, 0.05)
            noise_level = 0.05
            num_scenarios = min(num_scenarios, 5)
            click_range = (50, 150)
            zero_probability = 0.02
        elif difficulty == "hard":
            conversion_diff_range = (-0.01, 0.01)
            noise_level = 0.15
            click_range = (0, 250)
            zero_probability = 0.1

        time_windows = self.generate_time_windows(num_days)

        # --- select scenario names using Python random (seeded in __init__) ---
        if num_scenarios <= len(self.SCENARIO_NAMES):
            selected_names = random.sample(self.SCENARIO_NAMES, num_scenarios)
        else:
            selected_names = list(self.SCENARIO_NAMES)
            extra_count = num_scenarios - len(self.SCENARIO_NAMES)
            for i in range(extra_count):
                selected_names.append(f"Scenario_{len(self.SCENARIO_NAMES) + i + 1}")
            print(f"   ℹ️  Generated {extra_count} extra scenario names (Scenario_N format)")

        # --- derive per-scenario params from a master numpy RNG ---
        master_rng = np.random.RandomState(self._np_seed)
        scenario_seeds = master_rng.randint(0, 2**31, size=num_scenarios)

        # Pre-compute per-scenario random params from master RNG (sequential,
        # only 2 floats per scenario — negligible time even for 5000)
        base_conversions = master_rng.uniform(
            base_conversion_range[0], base_conversion_range[1], size=num_scenarios)
        conversion_diffs = master_rng.uniform(
            conversion_diff_range[0], conversion_diff_range[1], size=num_scenarios)

        def _generate_one(idx):
            rng = np.random.RandomState(int(scenario_seeds[idx]))
            arrays = self.generate_ab_data(
                time_windows=time_windows,
                base_conversion_rate=float(base_conversions[idx]),
                conversion_diff=float(conversion_diffs[idx]),
                click_range=click_range,
                noise_level=noise_level,
                zero_probability=zero_probability,
                rng=rng,
            )
            a_rate, b_rate = self.calculate_conversion_rate(arrays)
            return {
                "name": selected_names[idx],
                "arrays": arrays,
                "a_conversion_rate": a_rate,
                "b_conversion_rate": b_rate,
                "num_rows": len(time_windows),
            }

        # Parallel generation — numpy releases the GIL during heavy ops
        max_workers = min(8, num_scenarios)
        if num_scenarios > 50:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                scenarios = list(pool.map(_generate_one, range(num_scenarios)))
        else:
            scenarios = [_generate_one(i) for i in range(num_scenarios)]

        return {
            "scenarios": scenarios,
            "num_scenarios": len(scenarios),
            "num_days": num_days,
            "difficulty": difficulty,
            "parameters": {
                "base_conversion_range": base_conversion_range,
                "conversion_diff_range": conversion_diff_range,
                "click_range": click_range,
                "noise_level": noise_level,
                "zero_probability": zero_probability,
            }
        }

    def save_expected_ratio(self, scenarios: List[Dict], output_file: Path):
        """Save ground-truth conversion rates (works with numpy arrays)."""
        total_a_clicks = 0
        total_a_views = 0
        total_b_clicks = 0
        total_b_views = 0
        for s in scenarios:
            arr = s["arrays"]
            total_a_clicks += int(arr["A_clicks"].sum())
            total_a_views += int(arr["A_store_views"].sum())
            total_b_clicks += int(arr["B_clicks"].sum())
            total_b_views += int(arr["B_store_views"].sum())

        overall_a_rate = total_a_views / total_a_clicks if total_a_clicks > 0 else 0
        overall_b_rate = total_b_views / total_b_clicks if total_b_clicks > 0 else 0

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "A_conversion %", "B_conversion %"])

            for scenario in scenarios:
                writer.writerow([
                    scenario["name"],
                    f"{scenario['a_conversion_rate'] * 100:.3f}%",
                    f"{scenario['b_conversion_rate'] * 100:.3f}%"
                ])

            writer.writerow([
                "overall (total_store_views/total_clicks)",
                f"{overall_a_rate * 100:.3f}%",
                f"{overall_b_rate * 100:.3f}%"
            ])


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Generate A/B test data')

    parser.add_argument('--num-scenarios', type=int, default=20)
    parser.add_argument('--num-days', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='files')
    parser.add_argument('--difficulty', type=str, default='medium',
                        choices=['easy', 'medium', 'hard'])
    parser.add_argument('--base-conversion-min', type=float, default=0.70)
    parser.add_argument('--base-conversion-max', type=float, default=0.78)
    parser.add_argument('--conversion-diff-min', type=float, default=-0.03)
    parser.add_argument('--conversion-diff-max', type=float, default=0.03)
    parser.add_argument('--click-min', type=int, default=0)
    parser.add_argument('--click-max', type=int, default=200)
    parser.add_argument('--noise-level', type=float, default=0.1)
    parser.add_argument('--zero-probability', type=float, default=0.05)
    parser.add_argument('--save-groundtruth', action='store_true')
    parser.add_argument('--groundtruth-dir', type=str, default='groundtruth_workspace')

    args = parser.parse_args()

    print("=" * 60)
    print("A/B Test Data Generator (numpy-vectorized)")
    print("=" * 60)
    print(f"Scenarios: {args.num_scenarios}")
    print(f"Days per scenario: {args.num_days}")
    print(f"Rows per scenario: {args.num_days * 24}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    generator = ABTestingDataGenerator(seed=args.seed)

    result = generator.generate_scenarios(
        num_scenarios=args.num_scenarios,
        num_days=args.num_days,
        base_conversion_range=(args.base_conversion_min, args.base_conversion_max),
        conversion_diff_range=(args.conversion_diff_min, args.conversion_diff_max),
        click_range=(args.click_min, args.click_max),
        noise_level=args.noise_level,
        zero_probability=args.zero_probability,
        difficulty=args.difficulty,
    )

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        for old_file in output_dir.glob("ab_*.csv"):
            old_file.unlink()
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nWriting {len(result['scenarios'])} CSV files...")
    for scenario in result["scenarios"]:
        filename = f"ab_{scenario['name']}.csv"
        generator.save_csv(scenario["arrays"], output_dir / filename)

    if args.save_groundtruth:
        gt_dir = Path(args.groundtruth_dir)
        gt_dir.mkdir(exist_ok=True, parents=True)
        generator.save_expected_ratio(result["scenarios"], gt_dir / "expected_ratio.csv")
        print(f"Ground truth saved to {gt_dir / 'expected_ratio.csv'}")

    print(f"\nDone! {result['num_scenarios']} scenarios, "
          f"{result['num_scenarios'] * result['num_days'] * 24} total rows")


if __name__ == "__main__":
    main()
