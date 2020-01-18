using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.IO;
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
using System.Xml;

namespace Egomotion
{
    /// <summary>
    /// Logika interakcji dla klasy UserControl1.xaml
    /// </summary>
    public partial class Calibrator : UserControl
    {
        List<Emgu.CV.Image<Emgu.CV.Structure.Bgr, Byte>> imageList;
        List<System.Drawing.PointF[]> allPonits;
        Mat camMat;
        Emgu.CV.Util.VectorOfFloat distCoeffs;

        public Calibrator()
        {
            InitializeComponent();
        }

        private void LoadImage(object sender, RoutedEventArgs e)
        {
            imageList = new List<Emgu.CV.Image<Emgu.CV.Structure.Bgr, byte>>();
            FileOp.OpenFolder((dir) =>
            {
                var images = Directory.EnumerateFiles(dir);
                
                foreach (string s in images) {
                    var mat = Emgu.CV.CvInvoke.Imread(s, Emgu.CV.CvEnum.ImreadModes.Color);
                    if (mat != null)
                    {
                        imageList.Add(mat.ToImage<Bgr, byte>());
                    }

                }
            });
        }

        private void Circles(object sender, RoutedEventArgs e) 
        {
            allPonits = new List<System.Drawing.PointF[]>();
            foreach (var i in imageList)
            {
                Emgu.CV.Features2D.SimpleBlobDetectorParams simpleBlobDetectorParams = new Emgu.CV.Features2D.SimpleBlobDetectorParams();
                simpleBlobDetectorParams.MaxArea = 300000;
                simpleBlobDetectorParams.FilterByColor = true;
                simpleBlobDetectorParams.MinThreshold = 100;
                simpleBlobDetectorParams.MaxThreshold = 150;
                simpleBlobDetectorParams.ThresholdStep = 10;
                simpleBlobDetectorParams.MinArea = 1000;
                simpleBlobDetectorParams.FilterByArea = true;
                simpleBlobDetectorParams.MinDistBetweenBlobs = 10;
                var blobs = new Emgu.CV.Features2D.SimpleBlobDetector(simpleBlobDetectorParams).Detect(i);
                var points = Emgu.CV.CvInvoke.FindCirclesGrid(i.Convert<Gray, byte>(), new System.Drawing.Size(4,11), (Emgu.CV.CvEnum.CalibCgType)6, new Emgu.CV.Features2D.SimpleBlobDetector(simpleBlobDetectorParams));
                if(points != null)
                {
                    allPonits.Add(points);
                }
            }

            calcPosition();
        }

        private void calcPosition()
        {
            camMat = new Mat(3, 3, Emgu.CV.CvEnum.DepthType.Cv64F, 1);
            distCoeffs = new Emgu.CV.Util.VectorOfFloat();
            MCvPoint3D32f[][] mCvPoint3D32F = new MCvPoint3D32f[allPonits.Count][];

            for (int l = 0; l < allPonits.Count; l++)
            {
                mCvPoint3D32F[l] = new MCvPoint3D32f[44];
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 11; j++)
                    {
                        int pos =( i + j*4);
                        mCvPoint3D32F[l][pos] = new MCvPoint3D32f((10 - j) * 35.0f / 2.0f, i * 35.0f + (j % 2) * 35.0f / 2.0f, 0.0f);
                    }
                }
            }
            var sizeImg = imageList.First().Size;

            double err = Emgu.CV.CvInvoke.CalibrateCamera( mCvPoint3D32F, allPonits.ToArray(), sizeImg,camMat, distCoeffs, Emgu.CV.CvEnum.CalibType.Default, new MCvTermCriteria(100, 0.001), out Mat[] R, out Mat[] T);

            using(Stream file = new FileStream("cammat.txt", FileMode.OpenOrCreate))
            {
                SaveAndLoad.SaveCalibration(file, camMat, distCoeffs);
            }
            
        }

        private void Undistort(object sender, RoutedEventArgs e)
        {
            var image = ImageLoader.FromFile();
            if (image != null)
            {
                var dimage = image.Clone();
                var map1 = new Mat();
                var map2 = new Mat();
                Emgu.CV.CvInvoke.InitUndistortRectifyMap(camMat, distCoeffs, new Mat(), camMat, image.Size, Emgu.CV.CvEnum.DepthType.Cv32F, map1, map2);
                Emgu.CV.CvInvoke.Remap(image, dimage, map1, map2, Emgu.CV.CvEnum.Inter.Cubic, Emgu.CV.CvEnum.BorderType.Constant);

                imageViewer1.Source = ImageLoader.ImageSourceForBitmap(image.Bitmap);
                imageViewer2.Source = ImageLoader.ImageSourceForBitmap(dimage.Bitmap);
            }
        }
    }
}
