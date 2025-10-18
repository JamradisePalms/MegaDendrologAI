import 'package:flutter/material.dart';
import 'dart:io';
import '../models/report.dart';
import '../services/connectivity_service.dart';
import 'crop_screen.dart';
import 'package:flutter/foundation.dart'; // для consolidateHttpClientResponseBytes
import 'package:path_provider/path_provider.dart'; // для getTemporaryDirectory
import '../services/report_service.dart';
class ReportScreen extends StatefulWidget {
  final List<Report> reports;

  const ReportScreen({super.key, required this.reports});

  @override
  State<ReportScreen> createState() => _ReportScreenState();
}

class _ReportScreenState extends State<ReportScreen> {
  late final PageController _pageController;
  int _currentPage = 0;

  @override
  void initState() {
    super.initState();
    _pageController = PageController();
  }

  void _nextPage() {
    if (_currentPage < widget.reports.length - 1) {
      _pageController.nextPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    }
  }

  void _prevPage() {
    if (_currentPage > 0) {
      _pageController.previousPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    }
  }

  Future<Widget> _resolveImage(Report report) async {
    final url = report.imageUrl;
    final path = report.imagePath;
    final hasInternet = await ConnectivityService.hasInternet();

    if (hasInternet && url != null && url.isNotEmpty) {
      return Image.network(
        url,
        fit: BoxFit.contain,
        errorBuilder: (_, __, ___) => const Icon(Icons.broken_image, size: 200),
      );
    } else if (path != null && path.isNotEmpty) {
      if (path.startsWith('assets/')) {
        return Image.asset(path, fit: BoxFit.contain);
      } else {
        return Image.file(File(path), fit: BoxFit.contain);
      }
    } else {
      return const Icon(Icons.image_not_supported, size: 200);
    }
  }

  Widget _buildImage(Report report) {
    return FutureBuilder<Widget>(
      future: _resolveImage(report),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const CircularProgressIndicator();
        } else if (snapshot.hasError) {
          return const Icon(Icons.error, size: 200);
        } else {
          return Stack(
            alignment: Alignment.topRight,
            children: [
              ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 800, maxHeight: 600),
                child: snapshot.data!,
              ),
              Positioned(
                top: 8,
                right: 8,
                child: IconButton(
                  icon: const Icon(Icons.crop, color: Colors.white),
                  style: IconButton.styleFrom(
                    backgroundColor: Colors.black54,
                  ),
                  onPressed: () async {
                    final hasInternet = await ConnectivityService.hasInternet();

                    // Если есть интернет и URL доступен — скачиваем изображение и сохраняем локально
                    if (hasInternet &&
                        report.imageUrl != null &&
                        report.imageUrl!.isNotEmpty) {
                      try {
                        final httpClient = HttpClient();
                        final request = await httpClient.getUrl(Uri.parse(report.imageUrl!));
                        final response = await request.close();

                        if (response.statusCode == 200) {
                          final bytes = await consolidateHttpClientResponseBytes(response);

                          // путь для сохранения во временную папку
                          final dir = await getApplicationDocumentsDirectory();
                          final filePath =
                              '${dir.path}/report_${report.id ?? DateTime.now().millisecondsSinceEpoch}.jpg';
                          final file = File(filePath);
                          await file.writeAsBytes(bytes);

                          // сохраняем локальный путь в объект
                          report.imagePath = filePath;

                          // затем открываем CropScreen с локальным файлом
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => CropScreen(imageFile: file),
                            ),
                          );
                          return;
                        } else {
                          throw Exception('Ошибка загрузки изображения');
                        }
                      } catch (e) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(content: Text('Ошибка загрузки изображения: $e')),
                        );
                        return;
                      }
                    }

                    // Если путь уже есть — открываем кроп из локального файла
                    final path = report.imagePath;
                    if (path != null && path.isNotEmpty) {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => CropScreen(imageFile: File(path)),
                        ),
                      );
                    } else {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Изображение недоступно')),
                      );
                    }
                  },
                ),
              ),
            ],
          );
        }
      },
    );
  }



Widget _buildTable(Report report) {
  final reportService = ReportService();

  return Padding(
    padding: const EdgeInsets.all(16.0),
    child: DefaultTextStyle(
      style: const TextStyle(fontSize: 14, color: Colors.black87, height: 1.3),
      child: Container(
        decoration: BoxDecoration(
          color: const Color.fromARGB(0, 255, 255, 255),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.black, width: 1),
        ),
        clipBehavior: Clip.hardEdge,
        child: Table(
          border: const TableBorder.symmetric(
            inside: BorderSide(color: Colors.black, width: 1),
          ),
          defaultVerticalAlignment: TableCellVerticalAlignment.middle,
          columnWidths: const {0: FixedColumnWidth(180)},
          children: [
            _buildEditableRowGeneric(
              label: 'Название',
              value: report.plantName,
              onSave: (newValue) async {
                report.plantName = newValue;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Вероятность детекции растения',
              value: report.probability?.toString(),
              onSave: (newValue) async {
                report.probability = double.tryParse(newValue) ?? 0;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Общее состояние',
              value: report.overallCondition,
              onSave: (newValue) async {
                report.overallCondition = newValue;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Вид растения',
              value: report.species,
              onSave: (newValue) async {
                report.species = newValue;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Стволовые гнили',
              value: report.trunkRot,
              onSave: (newValue) async {
                report.trunkRot = newValue;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Дупла на стволе',
              value: report.trunkHoles,
              onSave: (newValue) async {
                report.trunkHoles = newValue;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Трещины на стволе',
              value: report.trunkCracks,
              onSave: (newValue) async {
                report.trunkCracks = newValue;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Повреждения ствола',
              value: report.trunkDamage,
              onSave: (newValue) async {
                report.trunkDamage = newValue;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Повреждения кроны',
              value: report.crownDamage,
              onSave: (newValue) async {
                report.crownDamage = newValue;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Плодовые тела',
              value: report.fruitingBodies,
              onSave: (newValue) async {
                report.fruitingBodies = newValue;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Болезни',
              value: report.diseases,
              onSave: (newValue) async {
                report.diseases = newValue;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Процент сухих ветвей',
              value: report.dryBranchPercentage?.toString(),
              onSave: (newValue) async {
                report.dryBranchPercentage = double.tryParse(newValue) ?? 0;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Геоданные',
              value: report.geoData,
              onSave: (newValue) async {
                report.geoData = newValue;
                await reportService.saveReport(report);
              },
            ),
            _buildEditableRowGeneric(
              label: 'Дополнительно',
              value: report.additionalInfo,
              onSave: (newValue) async {
                report.additionalInfo = newValue;
                await reportService.saveReport(report);
              },
            ),
          ],
        ),
      ),
    ),
  );
}






  // 👇 Добавляем поддержку редактирования additionalInfo
      // Универсальный метод для редактируемой строки
TableRow _buildEditableRowGeneric({
  required String label,
  required String? value,
  required void Function(String newValue) onSave,
}) {
  final controller = TextEditingController(text: value ?? '');
  bool isEditing = false;

  return TableRow(
    children: [
      TableCell(
        verticalAlignment: TableCellVerticalAlignment.middle,
        child: Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text(label, style: const TextStyle(fontWeight: FontWeight.bold)),
        ),
      ),
      TableCell(
        verticalAlignment: TableCellVerticalAlignment.middle,
        child: StatefulBuilder(
          builder: (context, setStateSB) {
            return Padding(
              padding: const EdgeInsets.all(8.0),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Expanded(
                    child: isEditing
                        ? TextField(
                            controller: controller,
                            maxLines: null,
                            textAlign: TextAlign.left,
                            style: const TextStyle(fontSize: 14, color: Colors.black87),
                            decoration: const InputDecoration(
                              hintText: 'Введите текст...',
                              border: OutlineInputBorder(),
                              isDense: true,
                              contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                            ),
                          )
                        : Text(
                            controller.text.isNotEmpty ? controller.text : 'Неизвестно',
                            textAlign: TextAlign.left,
                          ),
                  ),
                  IconButton(
                    icon: Icon(isEditing ? Icons.check : Icons.edit, color: Colors.black87),
                    onPressed: () async {
                      if (isEditing) {
                        // Сохраняем изменения через callback
                        onSave(controller.text.trim());
                      }
                      setStateSB(() => isEditing = !isEditing);
                    },
                  ),
                ],
              ),
            );
          },
        ),
      ),
    ],
  );
}





  TableRow _buildRow(String label, String value) {
    return TableRow(
      children: [
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text(label, style: const TextStyle(fontWeight: FontWeight.bold)),
        ),
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text(value),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Отчёты')),
      body: Column(
        children: [
          Expanded(
            child: PageView.builder(
              controller: _pageController,
              itemCount: widget.reports.length,
              onPageChanged: (index) => setState(() => _currentPage = index),
              itemBuilder: (context, index) {
                final report = widget.reports[index];
                return SingleChildScrollView(
                  child: Column(
                    children: [
                      _buildImage(report),
                      const SizedBox(height: 16),
                      _buildTable(report),
                    ],
                  ),
                );
              },
            ),
          ),
          // 👇 Панель управления страницами
          Container(
            color: Colors.grey.shade200,
            padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                IconButton(
                  onPressed: _prevPage,
                  icon: const Icon(Icons.arrow_back),
                ),
                Text('${_currentPage + 1} / ${widget.reports.length}'),
                IconButton(
                  onPressed: _nextPage,
                  icon: const Icon(Icons.arrow_forward),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
