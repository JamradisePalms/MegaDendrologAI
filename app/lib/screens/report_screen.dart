import 'package:flutter/material.dart';
import 'dart:io';
import '../models/report.dart';
import '../services/connectivity_service.dart';

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
          return ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 800, maxHeight: 600),
            child: snapshot.data!,
          );
        }
      },
    );
  }

  Widget _buildTable(Report report) {
    return Padding(
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
            report.dryBranchPercentage != null ? '${report.dryBranchPercentage}%' : 'неизвестно',
          ),
          _buildRow('Дополнительно', report.additionalInfo ?? 'неизвестно'),
        ],
      ),
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
