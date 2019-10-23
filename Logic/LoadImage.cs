﻿using Emgu.CV.Structure;
using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Egomotion
{
    public class ImageLoader
    {
        [DllImport("gdi32.dll", EntryPoint = "DeleteObject")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool DeleteObject([In] IntPtr hObject);

        public static ImageSource ImageSourceForBitmap(Bitmap bmp)
        {
            var handle = bmp.GetHbitmap();
            try
            {
                ImageSource newSource = Imaging.CreateBitmapSourceFromHBitmap(handle, IntPtr.Zero, Int32Rect.Empty, BitmapSizeOptions.FromEmptyOptions());

                DeleteObject(handle);
                return newSource;
            }
            catch (Exception)
            {
                DeleteObject(handle);
                return null;
            }
        }

        public static Emgu.CV.Image<Bgr, byte> FromFile()
        {
            Emgu.CV.Image<Bgr, byte> loadedImage = null;
            FileOp.LoadFromFile((s, path) =>
            {
                var mat = Emgu.CV.CvInvoke.Imread(path, Emgu.CV.CvEnum.ImreadModes.Color);
                if(mat != null)
                {
                    loadedImage = mat.ToImage<Bgr, byte>();
                }
            });
            return loadedImage;
        }
    }
}
