//
//  Deck.swift
//  ios_ws
//
//  Created by Bill on 11/10/15.
//  Copyright © 2015 An Yang. All rights reserved.
//

import Foundation

class Deck {
    
    static var rankString: [String] = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    var cards: [Card] = []
    
    init() {
        
        for rank in Deck.rankString {
            for suit in validSuit() {
                
                let card = Card()
                card.rank = rank
                card.suit = suit
                
                cards.append(card)
                
            }
        }
    }
    
    func addCard(card: Card) {
        
        self.cards.append(card)
    
    }
    
    func drawRandomCard() -> Card {
        
       
        let index = Int(arc4random() % UInt32(self.cards.count))
        
        let randomCard = self.cards[index]
        
        self.cards.remove(at:index)
        
        return randomCard
    }
    
    func validSuit() -> [String] {
    
        return ["♠", "♣", "♥", "♦"]
    }
    
}
