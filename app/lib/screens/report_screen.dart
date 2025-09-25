import 'package:flutter/material.dart';
import '../models/report.dart';
import 'dart:io';

class ReportScreen extends StatelessWidget {
  final Report report;

  const ReportScreen({super.key, required this.report});

  Widget _buildImage() {
    final path = report.imagePath;
    final url = report.imageUrl;

    Widget imageWidget;

    if (path != null && path.isNotEmpty) {
      // Локальный файл или assets
      imageWidget = path.startsWith('assets/')
          ? Image.asset(path, fit: BoxFit.contain)
          : Image.file(File(path), fit: BoxFit.contain);
    } else if (url != null && url.isNotEmpty) {
      // Онлайн картинка
      imageWidget = Image.network(url, fit: BoxFit.contain);
    } else {
      // Фоллбек
      imageWidget = const Icon(Icons.image_not_supported, size: 200);
    }

    return ConstrainedBox(
      constraints: const BoxConstraints(
        maxWidth: 400,
        maxHeight: 300,
      ),
      child: imageWidget,
    );
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Отчет')),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Картинка сверху
            _buildImage(),
            const SizedBox(height: 16),
            // Таблица
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Table(
                columnWidths: const {0: FixedColumnWidth(150)},
                border: TableBorder.all(),
                children: [
                _buildRow('Название растения', report.plantName ?? 'неизвестно'),
                _buildRow('Вероятность', report.probability != null ? '${report.probability}%' : '0'),
                _buildRow('Общее состояние', report.overallCondition ?? 'неизвестно'),
                _buildRow('Вид растения', report.species ?? 'неизвестно'),
                _buildRow('Стволовые гнили', report.trunkRot ?? 'неизвестно'),
                _buildRow('Дупла на стволе', report.trunkHoles ?? 'неизвестно'),
                _buildRow('Трещины на стволе', report.trunkCracks ?? 'неизвестно'),
                _buildRow('Повреждения ствола', report.trunkDamage ?? 'неизвестно'),
                _buildRow('Повреждения кроны', report.crownDamage ?? 'неизвестно'),
                _buildRow('Плодовые тела', report.fruitingBodies ?? 'неизвестно'),
                _buildRow('Болезни', report.diseases ?? 'неизвестно'),
                _buildRow(
                    'Процент сухих ветвей',
                    report.dryBranchPercentage != null
                        ? '${report.dryBranchPercentage}%'
                        : 'неизвестно'),
                _buildRow('Дополнительно', report.additionalInfo ?? 'неизвестно'),
              ],

              ),
            ),
          ],
        ),
      ),
    );
  }

  TableRow _buildRow(String label, String value) {
    return TableRow(
      children: [
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text(label, style: TextStyle(fontWeight: FontWeight.bold)),
        ),
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text(value),
        ),
      ],
    );
  }
}
