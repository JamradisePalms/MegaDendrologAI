import 'package:flutter/material.dart';
import '../models/report.dart';

class ReportScreen extends StatelessWidget {
  final Report report;

  const ReportScreen({super.key, required this.report});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Отчет')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Table(
          columnWidths: {0: FixedColumnWidth(150)},
          border: TableBorder.all(),
          children: [
            _buildRow('Название растения', report.plantName),
            _buildRow('Вероятность', '${report.probability}%'),
            _buildRow('Порода', report.species),
            _buildRow('Стволовые гнили', report.trunkRot),
            _buildRow('Дупла на стволе', report.trunkHoles),
            _buildRow('Трещины на стволе', report.trunkCracks),
            _buildRow('Повреждения ствола', report.trunkDamage),
            _buildRow('Повреждения кроны', report.crownDamage),
            _buildRow('Плодовые тела', report.fruitingBodies),
            _buildRow('Болезни', report.diseases),
            _buildRow('Процент сухих ветвей', '${report.dryBranchPercentage}%'),
            _buildRow('Дополнительно', report.additionalInfo),
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
