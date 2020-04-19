//
//  Game.swift
//  ios_ws
//
//  Created by Bill on 11/10/15.
//  Copyright © 2015 An Yang. All rights reserved.
//

import Foundation

class Game {
    
    var cardStack: [String] = []
    var equation: String = ""
    
    
    
    func clearPreString() {
        
        equation = ""
    }
    
    func getEquation() -> String {
        
        return self.equation
    }
    
    func calculator(equation: String) -> Int {
        
        var result: NSNumber = 0
        
        SwiftTryCatch.try ({
            
            let expression = NSExpression(format: equation)
            
            result = expression.expressionValue(with: nil, context: nil) as! NSNumber
            }, catch: { (error) in
                
                //print("caught exception ...")
                
                result = NSNumber(value: -99999)
                
            }, finallyBlock: {
        })
        
        return Int(result);
        
    }
    
    func pushCard(card: String) {
        
        cardStack.append(card)
        equation += String(rankToInt(content:card))
    }
    
    func pushOperation(operation: String) {
        
        equation += operation
    }
    
    func rankToInt(content: String)  -> Int {
        
        // swift 3
        let rank = content.substring(from: content.index(content.startIndex, offsetBy:1))
        
        
        // for swift2
        //let rank = content.substringFromIndex(content.startIndex.advancedBy(1))
        
        // for xcode version (6.4)
        //let rank = content.substringFromIndex(advance(content.startIndex,1))
        
        let lookUpTable: [String] = Deck.rankString
        
        for index in 0...12 {
            
            if lookUpTable[index] == rank {
                var trueIndex = index
                trueIndex = trueIndex + 1;
                return trueIndex
            }
            
        }
        
        return -1
        
    }
}
