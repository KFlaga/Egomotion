using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace Egomotion
{
    public partial class ImageViewer : ContentControl
    {
        public Image Child
        {
            get 
            {
                return image;
            }
        }

        public ImageSource Source
        {
            get { return image.Source; }
            set { image.Source = value; }
        }

        private Point actualCenterNormalized = new Point(0, 0); // Ranges from -0.5 to 0.5
        private const double scrollBarSize = 20;

        private double scaledImageWidth => Child.ActualWidth * scale.ScaleX - this.ActualWidth + scrollBarSize;
        private double scaledImageHeight => Child.ActualHeight * scale.ScaleY - this.ActualHeight + scrollBarSize;

        public ImageViewer()
        {
            InitializeComponent();
        }

        private void Reset()
        {
            scale.ScaleX = 1.0;
            scale.ScaleY = 1.0;
            translation.X = 0.0;
            translation.Y = 0.0;
        }

        private void Zoom(double zoom)
        {
            scale.ScaleX += zoom;
            scale.ScaleY += zoom;

            // zoom in center of screen
            translation.X = translation.X * (scale.ScaleX / (scale.ScaleX - zoom));
            translation.Y = translation.Y * (scale.ScaleY / (scale.ScaleY - zoom));

            if (scaledImageHeight > 0) // Content is higher than screen
            {
                vScroll.IsEnabled = true;
                actualCenterNormalized.Y = translation.Y / scaledImageHeight;
            }
            else
            {
                translation.Y = 0;
                vScroll.IsEnabled = false;
            }
            if (scaledImageWidth > 0) // Content is wider than screen
            {
                hScroll.IsEnabled = true;
                actualCenterNormalized.X = translation.X / scaledImageWidth;
            }
            else
            {
                translation.X = 0;
                hScroll.IsEnabled = false;
            }
            
            UpdateScrollBars();
        }

        private void MoveCenterTo(Point newCenter)
        {
            actualCenterNormalized = newCenter;
            translation.X = actualCenterNormalized.X * scaledImageWidth;
            translation.Y = actualCenterNormalized.Y * scaledImageHeight;
        }

        private void UpdateScrollBars()
        {
            hScroll.Value = 0.5 - actualCenterNormalized.X;
            vScroll.Value = 0.5 - actualCenterNormalized.Y;
        }

        private void HScroll_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            MoveCenterTo(new Point(0.5 - e.NewValue, actualCenterNormalized.Y));
        }

        private void VScroll_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            MoveCenterTo(new Point(actualCenterNormalized.X, 0.5 - e.NewValue));
        }

        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            double zoom = e.Delta > 0 ? .2 : -.2;
            if (!(e.Delta > 0) && (scale.ScaleX < .4 || scale.ScaleY < .4))
                return;
            
            Zoom(zoom);
            base.OnMouseWheel(e);
        }

        protected override void OnMouseDoubleClick(MouseButtonEventArgs e)
        {
            Reset();
            base.OnMouseDoubleClick(e);
        }

        private Point mouseScrollLast = new Point();

        protected override void OnPreviewMouseRightButtonDown(MouseButtonEventArgs e)
        {
            mouseScrollLast = e.GetPosition(this);
            base.OnPreviewMouseRightButtonDown(e);
        }

        protected override void OnPreviewMouseMove(MouseEventArgs e)
        {
            if (e.RightButton == MouseButtonState.Pressed)
            {
                Point position = e.GetPosition(this);
                double xDiff = (mouseScrollLast.X - position.X) / (ActualWidth * scale.ScaleX);
                double yDiff = (mouseScrollLast.Y - position.Y) / (ActualHeight * scale.ScaleY);

                double clamp(double x, double min, double max) => Math.Max(min, Math.Min(max, x));

                actualCenterNormalized = new Point(
                    x: clamp(actualCenterNormalized.X - xDiff, -0.5, 0.5),
                    y: clamp(actualCenterNormalized.Y - yDiff, -0.5, 0.5)
                );
                UpdateScrollBars();

                mouseScrollLast = position;
            }
            base.OnPreviewMouseMove(e);
        }
    }
}
