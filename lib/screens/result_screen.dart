import 'dart:async';
import 'dart:convert';
import 'dart:ui' as ui;
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:http/http.dart' as http;
import 'package:audioplayers/audioplayers.dart';
import 'package:geolocator/geolocator.dart';

class FaceRecognitionScreen extends StatefulWidget {
  const FaceRecognitionScreen({super.key});

  @override
  _FaceRecognitionScreenState createState() => _FaceRecognitionScreenState();
}

class _FaceRecognitionScreenState extends State<FaceRecognitionScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  late Future<void> _initializeControllerFuture;
  bool recognizing = false;
  int cooldownMs = 3000;
  int lastRecognition = 0;

  String userId = "";
  String userName = "";
  String statusMsg = "Initializing camera...";
  String? imageUrl;
  Uint8List? capturedFaceBytes;
  final AudioPlayer _audioPlayer = AudioPlayer();

  late final FaceDetector _faceDetector;

  // Geofence
  final double geofenceLat = 12.997666;
  final double geofenceLng = 77.669542;
  final double geofenceRadiusM = 60.0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableContours: false,
        enableClassification: false,
        enableLandmarks: false,
        performanceMode: FaceDetectorMode.accurate,
      ),
    );
    _initCamera();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _controller?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    final frontCamera = cameras.firstWhere(
      (cam) => cam.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    _controller = CameraController(frontCamera, ResolutionPreset.high, enableAudio: false);
    _initializeControllerFuture = _controller!.initialize();
    await _initializeControllerFuture;

    if (!mounted) return;

    // Start image stream
    _controller!.startImageStream(_processCameraImage);
    setState(() => statusMsg = "Camera ready ✅");
  }

  Future<void> _processCameraImage(CameraImage cameraImage) async {
    if (recognizing) return;
    if (DateTime.now().millisecondsSinceEpoch - lastRecognition < cooldownMs) return;

    try {
      final bytes = _concatenatePlanes(cameraImage);

      final rotation = InputImageRotation.rotation0deg; // Portrait mode
      final inputImage = InputImage.fromBytes(
        bytes: bytes,
        metadata: InputImageMetadata(
          size: Size(cameraImage.width.toDouble(), cameraImage.height.toDouble()),
          rotation: rotation,
          format: InputImageFormatValue.fromRawValue(cameraImage.format.raw) ?? InputImageFormat.nv21,
          bytesPerRow: cameraImage.planes[0].bytesPerRow,
        ),
      );

      final faces = await _faceDetector.processImage(inputImage);
      if (faces.isNotEmpty) {
        recognizing = true;
        lastRecognition = DateTime.now().millisecondsSinceEpoch;

        // Pick largest face
        final face = faces.reduce(
            (a, b) => a.boundingBox.width * a.boundingBox.height > b.boundingBox.width * b.boundingBox.height ? a : b);

        final croppedBytes = await _cropFace(cameraImage, face.boundingBox);
        if (croppedBytes == null) {
          recognizing = false;
          return;
        }

        setState(() {
          capturedFaceBytes = croppedBytes;
          imageUrl = null;
          statusMsg = "Sending for recognition...";
        });

        final base64Image = "data:image/jpeg;base64,${base64Encode(croppedBytes)}";
        await _sendToServer(base64Image);

        await Future.delayed(Duration(milliseconds: cooldownMs));
        recognizing = false;
      }
    } catch (e) {
      print("Error: $e");
    }
  }

  Uint8List _concatenatePlanes(CameraImage image) {
    final WriteBuffer allBytes = WriteBuffer();
    for (var plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    return allBytes.done().buffer.asUint8List();
  }

  Future<Uint8List?> _cropFace(CameraImage image, Rect boundingBox) async {
    try {
      final rgbBytes = _convertYUV420toRGB(image);

      final completer = Completer<ui.Image>();
      ui.decodeImageFromPixels(
        rgbBytes,
        image.width,
        image.height,
        ui.PixelFormat.rgba8888,
        (img) => completer.complete(img),
      );
      final fullImage = await completer.future;

      // Flip horizontally for front camera
      final recorder = ui.PictureRecorder();
      final canvas = Canvas(recorder);
      final paint = Paint();

      // Safe bounding box with margin
      final margin = 0.3 * max(boundingBox.width, boundingBox.height);
      int left = max(0, boundingBox.left - margin).toInt();
      int top = max(0, boundingBox.top - margin).toInt();
      int right = min(image.width, boundingBox.right + margin).toInt();
      int bottom = min(image.height, boundingBox.bottom + margin).toInt();

      int size = max(right - left, bottom - top);
      int cx = ((left + right) / 2).toInt();
      int cy = ((top + bottom) / 2).toInt();
      left = max(0, cx - size ~/ 2);
      top = max(0, cy - size ~/ 2);
      right = min(image.width, left + size);
      bottom = min(image.height, top + size);

      final srcRect = Rect.fromLTWH(left.toDouble(), top.toDouble(), (right - left).toDouble(), (bottom - top).toDouble());
      final dstRect = const Rect.fromLTWH(0, 0, 160, 160);

      // Mirror for front camera
      canvas.translate(160, 0);
      canvas.scale(-1, 1);
      canvas.drawImageRect(fullImage, srcRect, dstRect, paint);

      final picture = recorder.endRecording();
      final croppedImage = await picture.toImage(160, 160);
      final byteData = await croppedImage.toByteData(format: ui.ImageByteFormat.png);
      return byteData?.buffer.asUint8List();
    } catch (e) {
      print("Crop error: $e");
      return null;
    }
  }

  Uint8List _convertYUV420toRGB(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel!;
    final y = image.planes[0].bytes;
    final u = image.planes[1].bytes;
    final v = image.planes[2].bytes;

    final rgb = Uint8List(width * height * 4);
    int index = 0;

    for (int h = 0; h < height; h++) {
      final uvRow = uvRowStride * (h >> 1);
      for (int w = 0; w < width; w++) {
        final yp = y[h * width + w];
        final uvIndex = uvRow + (w >> 1) * uvPixelStride;
        final up = u[uvIndex];
        final vp = v[uvIndex];

        int r = (yp + 1.402 * (vp - 128)).toInt().clamp(0, 255);
        int g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128)).toInt().clamp(0, 255);
        int b = (yp + 1.772 * (up - 128)).toInt().clamp(0, 255);

        rgb[index++] = r;
        rgb[index++] = g;
        rgb[index++] = b;
        rgb[index++] = 255;
      }
    }
    return rgb;
  }

  bool _isInsideGeofence(double lat, double lng) {
    final distance = Geolocator.distanceBetween(lat, lng, geofenceLat, geofenceLng);
    return distance <= geofenceRadiusM;
  }

  Future<void> _sendToServer(String base64Image) async {
    try {
      final pos = await Geolocator.getCurrentPosition(desiredAccuracy: LocationAccuracy.high);
      if (!_isInsideGeofence(pos.latitude, pos.longitude)) {
        setState(() => statusMsg = "Outside Geofence ❌");
        _playSound("sounds/access_denied.mp3");
        return;
      }

      final response = await http.post(
        Uri.parse("http://10.96.117.221:5000/recognize"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "face_image": base64Image,
          "latitude": pos.latitude,
          "longitude": pos.longitude,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data["matched"] == true) {
          setState(() {
            userId = data["user_id"] ?? "";
            userName = data["name"] ?? "";
            imageUrl = data["image_url"];
            statusMsg = "Access Granted ✅";
          });
          _playSound("sounds/access_granted.mp3");
        } else {
          setState(() {
            statusMsg = data["message"] ?? "Face not recognized ❌";
            userId = "";
            userName = "";
            imageUrl = null;
          });
          _playSound("sounds/access_denied.mp3");
        }
      } else {
        setState(() => statusMsg = "Server error ❗");
      }
    } catch (e) {
      setState(() => statusMsg = "Network error ❗");
    }
  }

  void _playSound(String assetPath) async {
    await _audioPlayer.play(AssetSource(assetPath));
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    _audioPlayer.dispose();
    _faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: FutureBuilder(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return Stack(
              children: [
                Positioned.fill(child: CameraPreview(_controller!)), // full screen
                Positioned(
                  bottom: 30,
                  left: 20,
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      Container(
                        width: 90,
                        height: 90,
                        decoration: BoxDecoration(
                          color: Colors.black,
                          border: Border.all(color: Colors.white, width: 2),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        clipBehavior: Clip.hardEdge,
                        child: imageUrl != null
                            ? Image.network(
                                imageUrl!,
                                fit: BoxFit.cover,
                                errorBuilder: (_, __, ___) =>
                                    Image.asset("assets/images/placeholder.png", fit: BoxFit.cover),
                              )
                            : (capturedFaceBytes != null
                                ? Image.memory(capturedFaceBytes!, fit: BoxFit.cover)
                                : Image.asset("assets/images/placeholder.png", fit: BoxFit.cover)),
                      ),
                      const SizedBox(width: 10),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text("ID: $userId",
                              style: const TextStyle(
                                  color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold)),
                          Text("Name: $userName",
                              style: const TextStyle(
                                  color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold)),
                          const SizedBox(height: 4),
                          Text(statusMsg,
                              style: const TextStyle(
                                  color: Colors.yellowAccent, fontSize: 14, fontWeight: FontWeight.w500)),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            );
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
    );
  }
}
