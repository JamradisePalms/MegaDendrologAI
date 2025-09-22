import 'dart:io';
import 'dart:async';
import '../models/report.dart';

class ApiService {
  Future<Report> analyzeImage(File imageFile) async {
    // Имитируем задержку анализа
    await Future.delayed(Duration(seconds: 2));

    // Возвращаем фиктивный отчет
    return Report(
      plantName: 'Oak',
      probability: 95.0,
      species: 'Quercus robur',
      trunkRot: 'Нет',
      trunkHoles: 'Нет',
      trunkCracks: 'Мелкие',
      trunkDamage: 'Нет',
      crownDamage: 'Лёгкие',
      fruitingBodies: 'Есть',
      diseases: 'Отсутствуют',
      dryBranchPercentage: 10.0,
      additionalInfo: 'Здоровое дерево',
    );
  }
}
