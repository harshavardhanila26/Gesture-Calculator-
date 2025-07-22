Gesture Calculator 🖐️🧮
This project is a fully interactive virtual calculator that you can control using hand gestures. By tracking your hand with a webcam, it allows you to "press" buttons by pinching your fingers together, making for a fun, touchless interface.

The application is built with Python and uses OpenCV to render the UI on the live video feed and MediaPipe for high-fidelity hand tracking.

Features
Gesture-Based Input: Use a simple pinch gesture between your thumb and index finger to press buttons.

Real-Time Hand Tracking: Employs MediaPipe to accurately detect hand landmarks on the video feed.

Interactive UI: A complete calculator interface, including a display and buttons, is drawn directly onto the screen.

Vibrant Design: The calculator features round, rainbow-colored buttons for a lively and engaging user experience.

Full Functionality: Supports all basic arithmetic operations, a clear ('C') button, and an equals ('=') button that evaluates the expression.

Stable Controls: A built-in delay and cooldown system ensures that each pinch is registered as a single, intentional press.

How to Use
Run the script. A window will open showing your webcam feed with the calculator UI on the right side.

Aim with your index finger. Move your hand to position the tip of your index finger over the calculator button you want to press.

Pinch to Press. Bring your thumb and index finger close together to "press" the button. The calculation will appear in the display at the top.

Use the 'C' button to clear the current equation and '=' to see the final result.
