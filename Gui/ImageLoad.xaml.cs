using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Emgu.CV;
using Emgu.CV.Structure;

namespace Egomotion
{
    /// <summary>
    /// Logika interakcji dla klasy ImageLoad.xaml
    /// </summary>
    public partial class ImageLoad : UserControl
    {
        public ImageLoad()
        {
            InitializeComponent();
        }

        public ImageSource Source
        {
            get { return imageViewer.Source; }
            set { imageViewer.Source = value; }
        }

        public Image<Bgr, byte> loadedImage;

        private void LoadImage(object sender, RoutedEventArgs e)
        {

            loadedImage = ImageLoader.FromFile();
            if (loadedImage != null)
            {
                imageViewer.Source = ImageLoader.ImageSourceForBitmap(loadedImage.Bitmap);
            }
        }


    }
}
