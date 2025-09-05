"""
Flask web app for interactive visualization of the data.

Features:
- Ability to select features from the dataset to visualize (two at a time for scatter plots)
- Ability to color the points based on a categorical variable (e.g., treatment group)
- View the data as a scatter plot using d3.js
- Click on a point in the scatter plot to view the corresponding image from the dataset in a separate panel
"""

import pandas as pd
import numpy as np
import base64
import io
import matplotlib
import argparse

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify, request
import json

app = Flask(__name__)


class Data:
    """
    Backend for the data handling in the interactive visualization.

    Has access to the features in the form of a CSV file, and images from path.
    """

    def __init__(self, csv_path: str, folder: str):
        self.data = pd.read_csv(csv_path, index_col=0)
        # Set "ImageNumber" and "ObjectNumber" as index for easier access to images
        self.data.set_index(["ImageNumber", "ObjectNumber"], inplace=True)
        self.folder = folder
        self.feature_names = list(
            set(self.data.columns.tolist())
            - {
                "Metadata_Treatment",
                "Treatment_Binary",
            }
        )

    def get_image(self, image_number: int, object_number: int) -> str:
        """
        Returns a base64 encoded image for the given image number and object number.
        Currently returns a random numpy array as placeholder.
        """
        # Create a random image as placeholder
        random_image = np.random.rand(100, 100, 3)

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(random_image)
        ax.set_title(f"Image {image_number}, Object {object_number}")
        ax.axis("off")

        # Convert to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight")
        img_buffer.seek(0)
        img_string = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)

        return img_string

    def get_scatter_data(
        self, x_feature: str, y_feature: str, color_feature: str = None
    ):
        """
        Returns data formatted for d3.js scatter plot.
        """
        # Reset index to get ImageNumber and ObjectNumber as columns
        data_reset = self.data.reset_index()

        scatter_data = []
        for idx, row in data_reset.iterrows():
            point = {
                "x": float(row[x_feature]),
                "y": float(row[y_feature]),
                "imageNumber": int(row["ImageNumber"]),
                "objectNumber": int(row["ObjectNumber"]),
            }

            if color_feature and color_feature in row:
                point["color"] = str(row[color_feature])

            scatter_data.append(point)

        return scatter_data


@app.route("/")
def index():
    """Main page with the interactive visualization."""
    return render_template("index.html")


@app.route("/api/features")
def get_features():
    """API endpoint to get available features for dropdown menus."""
    return jsonify(
        {
            "features": data_handler.feature_names,
            "categorical_features": ["Metadata_Treatment", "Treatment_Binary"],
        }
    )


@app.route("/api/scatter_data")
def get_scatter_data():
    """API endpoint to get scatter plot data."""
    x_feature = request.args.get("x_feature")
    y_feature = request.args.get("y_feature")
    color_feature = request.args.get("color_feature")

    if not x_feature or not y_feature:
        return jsonify({"error": "x_feature and y_feature are required"}), 400

    try:
        scatter_data = data_handler.get_scatter_data(
            x_feature, y_feature, color_feature
        )
        return jsonify({"data": scatter_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/image")
def get_image():
    """API endpoint to get image for a specific data point."""
    image_number = request.args.get("image_number", type=int)
    object_number = request.args.get("object_number", type=int)

    if image_number is None or object_number is None:
        return jsonify({"error": "image_number and object_number are required"}), 400

    try:
        image_data = data_handler.get_image(image_number, object_number)
        return jsonify({"image": image_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Interactive visualization web app")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="joined_data_top10features.csv",
        help="Path to the CSV file containing the features data (default: joined_data_top10features.csv)",
    )
    parser.add_argument(
        "--images-folder",
        type=str,
        default="images/",
        help="Path to the folder containing images (default: images/)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port number to run the server on (default: 5001)",
    )
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")

    args = parser.parse_args()

    # Initialize data object with command line arguments
    data_handler = Data(args.csv_path, args.images_folder)

    print(f"Starting server with:")
    print(f"  CSV file: {args.csv_path}")
    print(f"  Images folder: {args.images_folder}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Debug mode: {args.debug}")

    app.run(debug=args.debug, host=args.host, port=args.port)
